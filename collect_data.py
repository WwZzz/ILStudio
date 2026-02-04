#!/usr/bin/env python
"""
collect_data.py

Data collection for teleoperation:
1. Start all devices (teleop optional, robot + cameras required)
2. Read from device SHM buffers, sync by timestamp (nearest-neighbor)
3. Enter to START recording, Enter to STOP, Enter to SAVE (or any input + Enter to DISCARD)
4. Save each episode as one HDF5 file

Usage:
    python collect_data.py --robot configs/robot/bi_so101_follower.yaml [--teleop configs/teleop/bi_so101_leader.yaml] -o data/teleop_recordings
    
    # With visualization (if robot module provides a Visualizer class):
    python collect_data.py --robot configs/robot/so101_sim_qpos.yaml --teleop configs/teleop/so101_leader.yaml --visualize
"""

# CRITICAL: Import shm_utils FIRST to patch resource_tracker before multiprocessing
import deploy.shm_utils  # noqa: F401 - applies _fix_resource_tracker() on import

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger
from deploy.base import start_device, is_robot_config, is_camera_config
from deploy.controller import KBHit
from deploy.shm_utils import SharedMemoryChannel, SharedMemoryDataSynchronizer, cleanup_all_shm
from deploy.utils import RateLimiter
from deploy.visualizer.base import start_all_visualizers, stop_all_visualizers
from configs.loader import ConfigLoader

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_config(path: str) -> List[dict]:
    """Load config from yaml. Returns list of device configs."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    return [raw]


def start_devices(configs: List[dict]) -> List[mp.Process]:
    """Start each device config in a separate process."""
    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        p.start()
        procs.append(p)
    return procs


def get_shm_name(cfg: dict) -> str:
    """Get SHM name from device config."""
    args = cfg.get("args", {})
    return args.get("name") or args.get("robot_id") or cfg.get("name", "unknown")


def _write_episode_hdf5(
    filepath: str,
    frames: List[Dict[str, dict]],
    device_names: List[str],
    teleop_device: Optional[str] = None,
) -> None:
    """
    Save episode frames to HDF5.
    Structure (lerobot-style):
      /observations/head_camera     - [T,H,W,C] images
      /observations/left_wrist_camera - [T,H,W,C]
      /observations/left_shoulder_pan.pos - [T] etc.
      /observations/qpos            - [T,12] if present
      /actions                     - [T, action_dim] from teleop (if present)
    """
    if not frames or not HAS_H5PY:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with h5py.File(filepath, "w") as f:
        obs_group = f.create_group("observations")

        def _extract_qpos(data: dict) -> Optional[np.ndarray]:
            """Extract qpos from observation (direct or from left/right *.pos)."""
            if "qpos" in data:
                return np.array(data["qpos"])
            left_keys = sorted(k for k in data if k.startswith("left_") and k.endswith(".pos"))
            right_keys = sorted(k for k in data if k.startswith("right_") and k.endswith(".pos"))
            if not left_keys and not right_keys:
                return None
            left = np.array([data[k] for k in left_keys], dtype=np.float32) if left_keys else np.array([])
            right = np.array([data[k] for k in right_keys], dtype=np.float32) if right_keys else np.array([])
            return np.concatenate([left, right])

        for dev_name in device_names:
            if dev_name not in frames[0]:
                continue
            data = frames[0][dev_name]
            if dev_name == teleop_device:
                continue  # teleop -> actions, not observations
            keys_to_save = [k for k in data if k not in ("__timestamp__", "timestamp")]
            if _extract_qpos(data) is not None and "qpos" not in keys_to_save:
                keys_to_save.append("qpos")
            for k in keys_to_save:
                if k in ("__timestamp__", "timestamp"):
                    continue
                if "image" in data and k == "image":
                    ds_name = dev_name
                else:
                    ds_name = k
                try:
                    if k == "qpos" and "qpos" not in data:
                        vals = [_extract_qpos(f[dev_name]) for f in frames if dev_name in f]
                        vals = [x for x in vals if x is not None]
                    else:
                        vals = [f[dev_name][k] for f in frames if dev_name in f and k in f[dev_name]]
                    if not vals:
                        continue
                    stacked = np.stack([np.array(x) for x in vals])
                    comp = "gzip" if stacked.nbytes > 10000 else None
                    obs_group.create_dataset(ds_name, data=stacked, compression=comp)
                except (ValueError, KeyError, TypeError):
                    pass
        if teleop_device and teleop_device in frames[0] and "action" in frames[0][teleop_device]:
            actions = []
            for fr in frames:
                if teleop_device in fr and "action" in fr[teleop_device]:
                    a = fr[teleop_device]["action"]
                    actions.append(np.array(a, dtype=np.float32).flatten())
            if actions:
                f.create_dataset("actions", data=np.stack(actions), compression="gzip")

        # Save sync timestamps
        ts_list = [fr.get("_sync_timestamp", 0.0) for fr in frames if "_sync_timestamp" in fr]
        if ts_list:
            f.create_dataset("timestamps", data=np.array(ts_list, dtype=np.float64))


def load_device_configs(args, unknown_args) -> Tuple[List[dict], List[str], List[dict]]:
    """
    Load device configs from args.
    
    Returns:
        Tuple of (all_configs, all_shm_names, teleop_configs)
    """
    # Build ConfigLoader with CLI overrides (supports --robot.xxx and --teleop.xxx)
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown_args)
    # Load configs using ConfigLoader (supports name or path)
    robot_configs = []
    teleop_configs = []
    
    # Load robot config
    try:
        robot_cfg, robot_path = cfg_loader.load_robot(args.robot)
        robot_configs = [robot_cfg] if not isinstance(robot_cfg, list) else robot_cfg
        logger.info("Loaded robot config from: {}", robot_path)
    except FileNotFoundError as e:
        # Fallback to direct yaml loading for backward compatibility
        logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
        robot_configs = load_config(args.robot)
    
    # Load teleop config
    if args.teleop.strip():
        try:
            teleop_cfg, teleop_path = cfg_loader.load_teleop(args.teleop)
            teleop_configs = [teleop_cfg] if not isinstance(teleop_cfg, list) else teleop_cfg
            logger.info("Loaded teleop config from: {}", teleop_path)
        except FileNotFoundError as e:
            # Fallback to direct yaml loading for backward compatibility
            logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
            teleop_configs = load_config(args.teleop)

    # Auto-fill control_shm_name for robot configs if teleop is specified
    if teleop_configs:
        # Get the first teleop's name as the control SHM source
        teleop_shm_name = get_shm_name(teleop_configs[0])
        
        for cfg in robot_configs:
            try:
                if is_robot_config(cfg):
                    args_dict = cfg.get('args', {})
                    # If control_shm_name is not specified, use teleop's name
                    if args_dict.get('control_shm_name') is None:
                        if 'args' not in cfg:
                            cfg['args'] = {}
                        cfg['args']['control_shm_name'] = teleop_shm_name
                        logger.info("Auto-set control_shm_name='{}' for robot '{}'", 
                                   teleop_shm_name, get_shm_name(cfg))
            except Exception as e:
                # Not a robot config or error checking, skip
                pass
    if teleop_configs:
        logger.info("Teleop devices: {}", [get_shm_name(c) for c in teleop_configs])
    logger.info("Robot devices: {}", [get_shm_name(c) for c in robot_configs])
    # Device configs to run: teleop (optional) + robot (all devices including cameras)
    all_configs = teleop_configs + robot_configs
    all_shm_names = [get_shm_name(c) for c in all_configs]
    return all_configs, all_shm_names, teleop_configs

def main():
    parser = argparse.ArgumentParser(description="Collect teleoperation data with timestamp sync")
    parser.add_argument("-r", "--robot", type=str, required=True, help="Robot config (name or path, e.g., 'so101_sim_qpos' or 'configs/robot/xxx.yaml')")
    parser.add_argument("-t", "--teleop", type=str, default="", help="Teleop config (name or path, optional)")
    parser.add_argument("-o", "--output-dir", type=str, default="data/teleop_recordings", help="Output directory for HDF5 episodes")
    parser.add_argument("-f", "--frequency", type=float, default=25.0, help="Target max Hz; actual = min(-f, slowest sensor)")
    parser.add_argument("-s", "--start-idx", type=int, default=0, help="Starting episode index")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable visualization (if robot module provides Visualizer)")
    args, unknown_args = parser.parse_known_args()

    all_configs, all_shm_names, teleop_configs = load_device_configs(args, unknown_args)

    # Clean orphaned SHM
    cleanup_all_shm(all_shm_names)
    logger.info("Cleaned up orphaned SHM segments.")

    # Start all devices
    logger.info("Starting all devices...")
    all_procs = start_devices(all_configs)
    time.sleep(2.0)
    
    # Start visualizer subprocesses if requested
    viz_procs = []
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=all_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
        )

    # Connect to all SHM channels
    shm_channels: List[Tuple[str, Optional[SharedMemoryChannel]]] = []
    for cfg in all_configs:
        name = get_shm_name(cfg)
        try:
            ch = SharedMemoryChannel(name, is_writer=False, timeout=10.0)
            shm_channels.append((name, ch))
            logger.info("Connected: {}", name)
        except Exception as e:
            logger.warning("Failed {}: {}", name, e)
            shm_channels.append((name, None))

    # Create synchronizer
    valid_channels = [(n, ch) for n, ch in shm_channels if ch is not None]
    if not valid_channels:
        logger.error("No SHM channels connected. Exiting.")
        for p in all_procs:
            p.terminate()
        return

    sync = SharedMemoryDataSynchronizer(
        shm_channels=valid_channels,
        buffer_maxlen=200,
        max_tolerance_s=0.05,
    )

    episode_idx = args.start_idx
    kb = KBHit()
    kb.set_curses_term()

    try:
        while True:
            # Wait for Enter to START
            logger.info("Press Enter to START episode {}...", episode_idx)
            while kb.get_input() is None:
                sync.read_and_buffer()
                time.sleep(0.01)
            # Consume any extra input
            while kb.get_input() is not None:
                pass

            # Recording loop: block until all devices have new data, save one frame, then rate-limit.
            # Actual rate = min(args.frequency, slowest sensor).
            logger.info("Recording episode {}. Press Enter to STOP...", episode_idx)
            frames: List[Dict[str, dict]] = []
            rate_limiter = RateLimiter()

            def stop_requested() -> bool:
                return kb.get_input() is not None

            recording_start_time = time.perf_counter()
            frame_count = 0
            last_debug_time = recording_start_time
            debug_interval = 2.0  # Print debug info every 2 seconds
            
            while not stop_requested():
                loop_start = time.perf_counter()
                frame = sync.get_synced_frame_blocking(stop_check=stop_requested, debug=False)
                sync_time = time.perf_counter()
                
                if frame is None:
                    break
                
                # Use actual recording time as sync reference (not device timestamps)
                # This fixes the issue where device timestamps may be stale
                ref_ts = time.perf_counter()
                frame["_sync_timestamp"] = ref_ts
                frames.append(frame)
                frame_count += 1
                
                # Debug output: show per-device data freshness
                if ref_ts - last_debug_time >= debug_interval:
                    # Calculate actual current fps
                    elapsed_since_start = ref_ts - recording_start_time
                    current_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
                    
                    # Show per-device info (device count and keys)
                    device_names = [k for k in frame.keys() if isinstance(frame.get(k), dict)]
                    
                    sync_delay_ms = (sync_time - loop_start) * 1000
                    logger.info("[DEBUG] Frame {}: fps={:.1f}, sync_wait={:.1f}ms, devices=[{}]",
                               frame_count, current_fps, sync_delay_ms, ", ".join(device_names))
                    last_debug_time = ref_ts
                
                rate_limiter.sleep(args.frequency)

            # Consume stop Enter
            while kb.get_input() is not None:
                pass

            recording_end_time = time.perf_counter()
            
            # Show actual frame rate (use actual wall-clock time)
            if frames:
                actual_elapsed = recording_end_time - recording_start_time
                actual_fps = len(frames) / actual_elapsed if actual_elapsed > 0 else 0
                logger.info("Episode {}: {} frames, {:.2f}s, {:.1f} Hz (target max {} Hz)", 
                           episode_idx, len(frames), actual_elapsed, actual_fps, args.frequency)

            # Save or discard
            logger.info("Save? (Enter=SAVE, any key+Enter=DISCARD)")
            prompt = None
            while prompt is None:
                prompt = kb.get_input()
                time.sleep(0.05)
            if len(prompt.strip()) == 0:
                if frames and HAS_H5PY:
                    out_path = os.path.join(args.output_dir, f"episode_{episode_idx:04d}.hdf5")
                    teleop_name = get_shm_name(teleop_configs[0]) if teleop_configs else None
                    _write_episode_hdf5(out_path, frames, [n for n, _ in valid_channels], teleop_device=teleop_name)
                    logger.success("Saved {} frames to {}", len(frames), out_path)
                    episode_idx += 1
                else:
                    logger.warning("No frames to save.")
            else:
                logger.info("Discarded.")

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        kb.set_normal_term()
        
        # Stop visualizers
        stop_all_visualizers(viz_procs)
        
        for _, ch in shm_channels:
            if ch is not None:
                try:
                    ch.destroy()
                except Exception:
                    pass
        for p in all_procs:
            p.terminate()
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
        cleanup_all_shm(all_shm_names)
        logger.info("Done.")


if __name__ == "__main__":
    main()
