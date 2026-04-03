#!/usr/bin/env python
"""
eval_real.py

Real robot evaluation with policy inference:
1. Start robot/camera devices in subprocesses (like collect_data.py)
2. Run inference process to produce action chunks
3. Use action manager to select and publish actions

Usage:
    # Local model evaluation
    python eval_real.py -r robot/so101_follower -m /path/to/checkpoint
    
    # Remote policy server evaluation
    python eval_real.py -r robot/so101_follower -m localhost:5000
    
    # Dummy policy for testing pipeline
    python eval_real.py -r robot/so101_follower -m __dummy-random_7x16
"""

# CRITICAL: Import shm_utils FIRST to patch resource_tracker before multiprocessing
import deploy.shm_utils  # noqa: F401 - applies _fix_resource_tracker() on import

import os
import yaml
import traceback
import time
import multiprocessing as mp
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from data_utils.utils import set_seed
from deploy.base import start_device, is_robot_config, is_camera_config
from deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm
from deploy.utils import RateLimiter
from deploy.action_manager import load_action_manager
from deploy.inference import start_inference_process, stop_inference_process
from deploy.visualizer.base import start_all_visualizers
from configs.loader import ConfigLoader


def _safe_name(s: str) -> str:
    """Sanitize a string for filesystem paths."""
    if s is None:
        return "none"
    s = str(s)
    for ch in ["/", "\\", " ", ":", ";", "|", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    return s


def _as_uint8_image(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert a numpy array to an image suitable for cv2.imwrite.
    Returns HxW, HxWx1, HxWx3, or HxWx4 numpy array, or None if not image-like.
    """
    if not isinstance(arr, np.ndarray):
        return None

    x = arr

    # Handle common channel-first formats
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))
    elif x.ndim == 4:
        # BCHW or BHWC -> save first dimension as separate images outside this function
        return None

    # Basic shape checks
    if x.ndim == 2:
        pass  # grayscale HxW
    elif x.ndim == 3 and x.shape[2] in (1, 3, 4):
        pass  # HxWxC
    else:
        return None

    # dtype normalization
    if np.issubdtype(x.dtype, np.floating):
        x_max = float(np.nanmax(x)) if x.size else 0.0
        if x_max <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    elif x.dtype == np.uint8:
        pass
    elif np.issubdtype(x.dtype, np.integer):
        # keep uint16 as-is (useful for depth), otherwise clamp to uint8
        if x.dtype == np.uint16:
            return x
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8, copy=False)

    return x


def _save_images_from_value(
    *,
    value,
    out_dir: str,
    frame_idx: int,
    device_name: str,
    key_path: str,
    color_order: str = "rgb",
) -> int:
    """
    Save image(s) found in `value` (np.ndarray or nested dict with 'image') to disk.
    Returns number of files saved.
    """
    import os
    import cv2
    saved = 0
    dev = _safe_name(device_name)
    key = _safe_name(key_path)

    # Nested dict case (common: {"image": ..., "timestamp": ...})
    if isinstance(value, dict):
        if "image" in value and isinstance(value["image"], np.ndarray):
            return _save_images_from_value(
                value=value["image"],
                out_dir=out_dir,
                frame_idx=frame_idx,
                device_name=device_name,
                key_path=f"{key_path}.image" if key_path else "image",
                color_order=color_order,
            )
        return 0

    if not isinstance(value, np.ndarray):
        return 0

    # Batch formats
    if value.ndim == 4:
        # Try BHWC
        if value.shape[-1] in (1, 3, 4):
            for b in range(value.shape[0]):
                img = _as_uint8_image(value[b])
                if img is None:
                    continue
                if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
                    # cv2.imwrite expects BGR/BGRA
                    if img.shape[2] == 3:
                        img = img[..., ::-1]
                    else:  # 4 channels: RGBA -> BGRA
                        img = img[..., [2, 1, 0, 3]]
                fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}_b{b:02d}.png")
                if cv2.imwrite(fp, img):
                    saved += 1
            return saved
        # Try BCHW
        if value.shape[1] in (1, 3, 4):
            for b in range(value.shape[0]):
                img = _as_uint8_image(value[b])
                if img is None:
                    continue
                if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
                    if img.shape[2] == 3:
                        img = img[..., ::-1]
                    else:
                        img = img[..., [2, 1, 0, 3]]
                fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}_b{b:02d}.png")
                if cv2.imwrite(fp, img):
                    saved += 1
            return saved
        return 0

    img = _as_uint8_image(value)
    if img is None:
        return 0

    if color_order == "rgb" and img.ndim == 3 and img.shape[2] in (3, 4):
        if img.shape[2] == 3:
            img = img[..., ::-1]
        else:
            img = img[..., [2, 1, 0, 3]]

    fp = os.path.join(out_dir, f"{frame_idx:06d}_{dev}_{key}.png")
    if cv2.imwrite(fp, img):
        saved += 1
    return saved


def save_synced_frame_images(*, synced_data: dict, out_dir: str, frame_idx: int, color_order: str = "rgb") -> int:
    """
    Save all image-like arrays inside a synced_data frame to `out_dir`.
    Returns number of images saved.
    """
    import os
    import cv2  # local import so non-camera runs won't require cv2 at import time

    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for dev_name, dev_data in (synced_data or {}).items():
        if not isinstance(dev_data, dict):
            continue

        # Heuristic: prefer common keys, but also scan everything image-shaped.
        for k, v in dev_data.items():
            k_lower = str(k).lower()
            likely_image = any(token in k_lower for token in ("image", "rgb", "color", "frame", "depth", "camera"))
            if isinstance(v, dict):
                saved += _save_images_from_value(
                    value=v,
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                    device_name=str(dev_name),
                    key_path=str(k),
                    color_order=color_order,
                )
            elif isinstance(v, np.ndarray):
                # If key doesn't look image-like, only save when array is clearly an image
                if likely_image or (
                    (v.ndim == 2)
                    or (v.ndim == 3 and (v.shape[-1] in (1, 3, 4) or v.shape[0] in (1, 3, 4)))
                    or (v.ndim == 4 and (v.shape[-1] in (1, 3, 4) or v.shape[1] in (1, 3, 4)))
                ):
                    saved += _save_images_from_value(
                        value=v,
                        out_dir=out_dir,
                        frame_idx=frame_idx,
                        device_name=str(dev_name),
                        key_path=str(k),
                        color_order=color_order,
                    )

        # Also handle known patterns like obs['head_camera']['image']
        for cam_key in ("head_camera", "left_wrist_camera", "right_wrist_camera", "front_camera"):
            if cam_key in dev_data and isinstance(dev_data[cam_key], dict) and "image" in dev_data[cam_key]:
                saved += _save_images_from_value(
                    value=dev_data[cam_key],
                    out_dir=out_dir,
                    frame_idx=frame_idx,
                    device_name=str(dev_name),
                    key_path=str(cam_key),
                    color_order=color_order,
                )

    return saved


def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model on real robot')
    
    # Robot configuration (supports name or path like collect_data.py)
    parser.add_argument('-r', '--robot_config', type=str, default='robot/so101_follower',
                       help='Robot config (name under configs/robot or path to yaml)')
    parser.add_argument('-pr', '--publish_rate', type=float, default=25,
                       help='Action publishing rate (Hz)')
    parser.add_argument('-sr', '--sensing_rate', type=float, default=20,
                       help='Sensing rate (Hz) - used by synchronizer')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    # Direct checkpoint loading
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='localhost:5000',
                       help='Path to model checkpoint OR server address (host:port) for remote policy server')
    parser.add_argument('-o', '--output_dir', type=str, default='',
                       help='Directory to save results (videos will be saved here)')
    parser.add_argument('-i', '--episode_id', type=int, default=-1,
                       help='Episode ID for video naming (auto-increment if -1)')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Image size for policy')

    # Debug: save obs images and action chunks to -o directory
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Save observation images and action chunks to output_dir for debugging.',
    )
    
    # Action manager
    parser.add_argument('-am', '--action_manager', type=str, default='basic',
                       help='Action manager config name or path to config file')
    
    # Visualization
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Enable visualization (if robot module provides Visualizer)')
    
    # Parse arguments (allow unknown for dotted overrides)
    args, unknown = parser.parse_known_args()
    
    # Store unknown args for ConfigLoader to process
    setattr(args, 'unknown_args', unknown)
    return args


def load_config(path: str) -> List[dict]:
    """Load config from yaml. Returns list of device configs."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    return [raw]


def get_shm_name(cfg: dict) -> str:
    """Get SHM name from device config."""
    args = cfg.get("args", {})
    return args.get("name") or args.get("robot_id") or cfg.get("name", "unknown")


def start_devices(configs: List[dict], daemon: bool = True) -> List[mp.Process]:
    """Start each device config in a separate process.
    
    Args:
        configs: List of device configurations
        daemon: If True, processes will be terminated when main process exits
                (including crashes/segfaults). Default True for safety.
    """
    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        p.daemon = daemon
        p.start()
        procs.append(p)
    return procs


def load_device_configs(args, unknown_args, cfg_loader=None) -> Tuple[List[dict], List[str], List[dict]]:
    """
    Load device configs from robot_config argument.
    Sets control_shm_name to 'policy_control_shm' for robot devices.
    Visualizer entries are stripped from the device list and returned separately.
    """
    from deploy.utils import partition_visualizer_configs, coerce_finalize_deploy_list

    if cfg_loader is None:
        cfg_loader = ConfigLoader(args=args, unknown_args=unknown_args)
    robot_configs = []
    
    # Load robot config
    try:
        robot_cfg, robot_path = cfg_loader.load_robot(args.robot_config)
        robot_configs = coerce_finalize_deploy_list(robot_cfg)
        logger.info("Loaded robot config from: {}", robot_path)
    except FileNotFoundError as e:
        # Fallback to direct yaml loading for backward compatibility
        logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
        robot_configs = coerce_finalize_deploy_list(load_config(args.robot_config))

    # Separate visualizer entries from real device entries
    robot_configs, visualizer_configs = partition_visualizer_configs(robot_configs)
    
    # Set control_shm_name to 'policy_control_shm' for robot devices
    # This connects robot to the policy's action output
    for cfg in robot_configs:
        try:
            if is_robot_config(cfg):
                if 'args' not in cfg:
                    cfg['args'] = {}
                cfg['args']['control_shm_name'] = 'policy_control_shm'
                logger.info("Set control_shm_name='policy_control_shm' for robot '{}'", 
                           get_shm_name(cfg))
        except Exception:
            pass
    
    logger.info("Robot devices: {}", [get_shm_name(c) for c in robot_configs])
    if visualizer_configs:
        logger.info("Visualizer entries: {}", [v.get('name') or v.get('type') for v in visualizer_configs])
    all_shm_names = [get_shm_name(c) for c in robot_configs]
    return robot_configs, all_shm_names, visualizer_configs


def main():
    import signal
    import atexit
    
    set_seed(0)
    args = parse_param()
    args.is_training = False
    
    # Global process tracking for cleanup
    _all_procs = []
    _all_shm = []
    _cleanup_done = False
    
    def cleanup_all():
        """Cleanup function to be called on exit."""
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        
        logger.info("Running cleanup...")
        
        # Terminate all tracked processes
        for p in _all_procs:
            if p.is_alive():
                p.terminate()
        
        # Wait for graceful shutdown
        for p in _all_procs:
            p.join(timeout=3.0)
            if p.is_alive():
                logger.warning("Process {} did not exit gracefully, killing...", p.pid)
                p.kill()
                p.join(timeout=1.0)
        
        # Cleanup SHM
        for shm in _all_shm:
            try:
                shm.destroy()
            except Exception:
                pass
        
        logger.info("Cleanup complete.")
    
    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM in main process."""
        logger.info("Received signal {}, cleaning up...", signum)
        cleanup_all()
        exit(0)
    
    # Register signal handlers and atexit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_all)
    
    # Create shared ConfigLoader instance
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    
    # Load device configs (visualizer entries are separated out)
    robot_configs, all_shm_names, visualizer_configs = load_device_configs(
        args, getattr(args, 'unknown_args', []), cfg_loader=cfg_loader
    )
    
    # Clean orphaned SHM
    shm_to_clean = all_shm_names + ['policy_control_shm']
    cleanup_all_shm(shm_to_clean)
    logger.info("Cleaned up orphaned SHM segments.")
    
    # Create policy control SHM (main process writes actions here)
    control_shm = SharedMemoryChannel(name='policy_control_shm', max_size_mb=1, is_writer=True)
    _all_shm.append(control_shm)
    logger.info("Created policy_control_shm for action publishing.")
    
    # Start all devices in subprocesses
    logger.info("Starting all devices...")
    device_procs = start_devices(robot_configs)
    _all_procs.extend(device_procs)
    time.sleep(2.0)
    
    # Start visualizer subprocesses if requested
    viz_procs = []
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=robot_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
            visualizer_configs=visualizer_configs,
        )
        _all_procs.extend(viz_procs)
    
    # Start inference process via deploy/inference.py (trigger-based)
    inference_ctx = start_inference_process(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        device_shm_names=all_shm_names,
        device_configs=robot_configs,
    )
    _all_procs.append(inference_ctx.process)
    
    # Load action manager and bind to inference context
    logger.info("Loading action manager: {}", args.action_manager)
    try:
        manager_cfg, manager_cfg_path = cfg_loader.load_manager(args.action_manager)
        logger.info("Loaded action manager config from: {}", manager_cfg_path)
    except Exception as e:
        logger.info("Using legacy loading for: {}", args.action_manager)
        manager_cfg = {'manager_name': args.action_manager}
    
    action_manager = load_action_manager(config=manager_cfg)
    action_manager.set_inference_context(inference_ctx)
    logger.info("ActionManager: {} (bound to InferenceContext)", action_manager.__class__.__name__)
    
    # Press Enter to start the control loop
    input("Press Enter to start the control loop...")
    
    # Main control loop
    logger.info("="*60)
    logger.info("[Main Control Loop] Started. Press Ctrl+C to stop.")
    logger.info("  Publish rate: {} Hz", args.publish_rate)
    logger.info("="*60)
    
    rate_limiter = RateLimiter()
    action_count = 0
    last_log_time = time.perf_counter()
    
    try:
        while True:
            t = time.perf_counter()
            
            # select_action() handles everything:
            # 1. Poll for new action chunks from inference subprocess
            # 2. If buffer empty or should_infer(): send trigger
            # 3. If buffer empty: block-wait for inference result
            # 4. Return single-step action, auto-increment t
            step = action_manager.select_action()
            
            # Extract the action array from the mact step
            if isinstance(step, np.ndarray) and step.dtype == object and len(step) > 0:
                action_arr = step[0].get('action') if isinstance(step[0], dict) else step[0]
            elif isinstance(step, dict):
                action_arr = step.get('action', step)
            else:
                action_arr = step
            
            if action_arr is not None:
                action_arr = np.asarray(action_arr)
                if action_arr.size > 0:
                    control_shm.write({
                        'action': action_arr,
                        'timestamp': t,
                    })
                    action_count += 1
            
            if t - last_log_time > 5.0:
                logger.info("[Control Loop] Published {} actions in last 5s ({:.1f} Hz)", 
                           action_count, action_count / 5.0)
                action_count = 0
                last_log_time = t
            
            rate_limiter.sleep(args.publish_rate)
            
    except KeyboardInterrupt:
        logger.info("[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
    except Exception as e:
        logger.info("[Main Control Loop] Error: {}", e)
        traceback.print_exc()
    finally:
        stop_inference_process(inference_ctx)
        cleanup_all()
        cleanup_all_shm(shm_to_clean)


if __name__ == '__main__':
    main()
