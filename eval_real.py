#!/usr/bin/env python
"""
eval_real.py

Real robot evaluation with policy inference:
1. Start robot/camera devices in subprocesses (like collect_data.py)
2. Start unified inference_worker (REAL mode) - reads device SHMs directly
3. Use action_manager to trigger inference and select actions

Architecture:
    ┌──────────────┐             ┌───────────────────┐
    │ Device SHMs   │──(direct)──>│ inference_worker   │
    │ (device procs)│             │  (REAL mode)       │
    └──────────────┘             │ sync+obs2meta+infer│
                                  └────────┬──────────┘
                                           │ action_shm
                                           ▼
    ┌──────────────┐  ctrl_shm   ┌───────────────────┐
    │ action_manager│──(trigger)─>│                   │
    │  (main proc)  │<───────────│                   │
    └──────┬───────┘             └───────────────────┘
           │ single-step action
           ▼
    ┌──────────────┐
    │ control_shm   │──> Robot
    └──────────────┘

Key: Main process does NOT touch obs. Inference worker reads device SHMs directly.

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
from deploy.visualizer.base import start_all_visualizers, stop_all_visualizers
from configs.loader import ConfigLoader


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
    parser.add_argument('-ir', '--infer_rate', type=float, default=10,
                       help='Inference rate (Hz)')
    
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
    
    # Action manager
    parser.add_argument('-am', '--action_manager', type=str, default='older_first',
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


def _device_spawns_children(cfg: dict) -> bool:
    """True if this device type spawns child processes (e.g. GUI), so it must not run in a daemon process."""
    device_type = cfg.get("type", "")
    return "keyboard" in device_type or "tk_slider" in device_type


def start_devices(configs: List[dict], daemon: bool = True) -> List[mp.Process]:
    """Start each device config in a separate process.
    
    Args:
        configs: List of device configurations
        daemon: If True, processes will be terminated when main process exits
                (including crashes/segfaults). Default True for safety.
                Devices that spawn children (e.g. Keyboard, TkSlider GUI) are always
                started with daemon=False so they can create subprocesses.
    """
    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        # Daemon processes cannot have children; Keyboard/TkSlider start GUI subprocesses
        p.daemon = False if _device_spawns_children(cfg) else daemon
        p.start()
        procs.append(p)
    return procs


def load_device_configs(args, unknown_args, cfg_loader=None) -> Tuple[List[dict], List[str]]:
    """
    Load device configs from robot_config argument.
    Sets control_shm_name to 'policy_control_shm' for robot devices.
    """
    if cfg_loader is None:
        cfg_loader = ConfigLoader(args=args, unknown_args=unknown_args)
    robot_configs = []
    
    # Load robot config
    try:
        robot_cfg, robot_path = cfg_loader.load_robot(args.robot_config)
        robot_configs = [robot_cfg] if not isinstance(robot_cfg, list) else robot_cfg
        logger.info("Loaded robot config from: {}", robot_path)
    except FileNotFoundError as e:
        # Fallback to direct yaml loading for backward compatibility
        logger.warning("ConfigLoader failed, falling back to direct YAML: {}", e)
        robot_configs = load_config(args.robot_config)
    
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
    all_shm_names = [get_shm_name(c) for c in robot_configs]
    return robot_configs, all_shm_names


def get_robot_module_info(robot_configs: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the robot module path and class name from the first robot config.
    Used for loading obs2meta function in inference process.
    
    Returns:
        Tuple of (module_name, class_name) or (None, None) if not found
    """
    for cfg in robot_configs:
        try:
            if is_robot_config(cfg):
                device_type = cfg['type']
                parts = device_type.rsplit('.', 1)
                device_module_name = parts[0]
                device_class_name = parts[1]
                return device_module_name, device_class_name
        except Exception as e:
            logger.warning("Failed to get robot info from {}: {}", cfg.get('type', 'unknown'), e)
    
    return None, None


def _extract_action_array(mact_step):
    """
    Extract a raw numpy action array from a mact_list element.
    
    mact_step is np.array([{"action": np.ndarray, ...}, ...], dtype=object).
    For real robot, we extract and return the first action array.
    """
    if mact_step is None:
        return None
    if isinstance(mact_step, np.ndarray) and mact_step.dtype == object:
        # mact_list element: array of dicts
        if len(mact_step) > 0 and isinstance(mact_step[0], dict):
            return mact_step[0].get('action', None)
    # Already a raw array
    if isinstance(mact_step, np.ndarray):
        return mact_step
    return None


def main():
    import signal
    import atexit
    
    set_seed(0)
    args = parse_param()
    args.is_training = False
    
    # Global tracking for cleanup
    _all_procs = []
    _all_shm = []
    _cleanup_done = False
    inference_ctx = None
    
    def cleanup_all():
        """Cleanup function to be called on exit."""
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        
        logger.info("Running cleanup...")
        
        # Stop inference process
        if inference_ctx is not None:
            try:
                stop_inference_process(inference_ctx)
            except Exception as e:
                logger.warning("Error stopping inference process: {}", e)
        
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
    
    # Load device configs
    robot_configs, all_shm_names = load_device_configs(
        args, getattr(args, 'unknown_args', []), cfg_loader=cfg_loader
    )
    
    # Get robot module info for obs2meta function in inference process
    robot_module_name, robot_class_name = get_robot_module_info(robot_configs)
    
    # Clean orphaned SHM
    shm_to_clean = all_shm_names + ['policy_control_shm']
    cleanup_all_shm(shm_to_clean)
    logger.info("Cleaned up orphaned SHM segments.")
    
    # Create policy control SHM (main process writes actions here for robot)
    control_shm = SharedMemoryChannel(name='policy_control_shm', max_size_mb=1, is_writer=True)
    _all_shm.append(control_shm)
    logger.info("Created policy_control_shm for action publishing.")
    
    # Start all devices in subprocesses
    logger.info("Starting all devices...")
    device_procs = start_devices(robot_configs)
    _all_procs.extend(device_procs)
    time.sleep(2.0)
    
    # Start visualizer subprocesses if requested
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=robot_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
        )
        _all_procs.extend(viz_procs)
    
    # Start unified inference process (REAL mode)
    # Worker reads device SHMs directly → no obs relay through main process
    logger.info("Starting inference process (REAL mode)...")
    inference_ctx = start_inference_process(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        device_shm_names=all_shm_names,
        robot_module_name=robot_module_name,
        sync_buffer_maxlen=100,
        sync_max_tolerance_s=0.05,
    )
    
    # Load action manager
    logger.info("Loading action manager: {}", args.action_manager)
    try:
        manager_cfg, manager_cfg_path = cfg_loader.load_manager(args.action_manager)
        logger.info("✓ Loaded config from: {}", manager_cfg_path)
        action_manager = load_action_manager(config=manager_cfg)
    except Exception:
        logger.info("Using direct loading for: {}", args.action_manager)
        action_manager = load_action_manager(args.action_manager)
    
    # Connect action_manager to inference process
    action_manager.set_inference_context(inference_ctx)
    logger.info("ActionManager: {} (config: {})", action_manager.__class__.__name__, args.action_manager)
    
    # Press Enter to start the control loop
    input("Press Enter to start the control loop...")
    
    # Main control loop
    logger.info("=" * 60)
    logger.info("[Main Control Loop] Started. Press Ctrl+C to stop.")
    logger.info("  Publish rate: {} Hz", args.publish_rate)
    logger.info("  Action manager: {}", action_manager.__class__.__name__)
    logger.info("=" * 60)
    
    rate_limiter = RateLimiter()
    action_count = 0
    last_log_time = time.perf_counter()
    
    try:
        while True:
            t = time.perf_counter()
            
            # Get action from action_manager (triggers inference as needed)
            # Main process does NOT handle obs — worker reads device SHMs directly
            # Step counter is managed internally by action_manager (action_manager.t)
            mact_step = action_manager.select_action()
            
            # Extract raw action array from mact_list element
            action_arr = _extract_action_array(mact_step)
            
            # Write action to control_shm for robot to read
            if action_arr is not None:
                # Validate
                if action_arr.size == 0:
                    continue
                if action_arr.dtype == object:
                    logger.warning("[Main] action has object dtype, skipping")
                    continue
                if action_arr.ndim == 0:
                    action_arr = action_arr.reshape(1)
                
                action_data = {
                    'action': action_arr,
                    'timestamp': t,
                }
                control_shm.write(action_data)
                action_count += 1
            
            # Log stats periodically
            if t - last_log_time > 5.0:
                logger.info("[Control Loop] Published {} actions in last 5s ({:.1f} Hz)",
                           action_count, action_count / 5.0)
                action_count = 0
                last_log_time = t
            
            rate_limiter.sleep(args.publish_rate)
    
    except KeyboardInterrupt:
        logger.info("[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
    except Exception as e:
        logger.error("[Main Control Loop] Error: {}", e)
        traceback.print_exc()
    finally:
        cleanup_all()
        cleanup_all_shm(shm_to_clean)


if __name__ == '__main__':
    main()
