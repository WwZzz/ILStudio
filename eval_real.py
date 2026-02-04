#!/usr/bin/env python
"""
eval_real.py

Real robot evaluation with policy inference:
1. Start robot/camera devices in subprocesses (like collect_data.py)
2. Run inference process to produce action chunks
3. Use action manager to select and publish actions

Usage:
    # Local model evaluation
    python eval_real.py -c robot/so101_follower -m /path/to/checkpoint
    
    # Remote policy server evaluation
    python eval_real.py -c robot/so101_follower -m localhost:5000
    
    # Dummy policy for testing pipeline
    python eval_real.py -c robot/so101_follower -m __dummy-random_7x16
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
import torch
from loguru import logger

from data_utils.utils import set_seed
from deploy.base import start_device, is_robot_config, is_camera_config
from deploy.shm_utils import SharedMemoryDataSynchronizer, SharedMemoryChannel, cleanup_all_shm
from deploy.utils import RateLimiter
from deploy.action_manager import load_action_manager
from deploy.visualizer.base import start_all_visualizers, stop_all_visualizers
from policy.utils import load_policy
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
    parser.add_argument('-c', '--robot_config', type=str, default='robot/so101_follower',
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


def start_devices(configs: List[dict]) -> List[mp.Process]:
    """Start each device config in a separate process."""
    procs = []
    for cfg in configs:
        p = mp.Process(target=start_device, args=(cfg,))
        p.start()
        procs.append(p)
    return procs


def load_device_configs(args, unknown_args) -> Tuple[List[dict], List[str]]:
    """
    Load device configs from robot_config argument.
    Sets control_shm_name to 'policy_control_shm' for robot devices.
    """
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
    import importlib
    
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


def inference_process(args, shm_names: List[str], robot_module_name: Optional[str], robot_class_name: Optional[str]):
    """
    Inference producer process.
    Consumes observation data from SHM and produces action chunks.
    """
    from deploy.shm_utils import _fix_resource_tracker
    _fix_resource_tracker()
    
    import importlib
    from benchmark.base import MetaObs
    
    # Load policy
    policy = load_policy(args)
    logger.info("[Inference Process] Policy loaded.")
    
    # Connect to device SHM channels
    shm_channels: List[Tuple[str, SharedMemoryChannel]] = []
    for name in shm_names:
        try:
            ch = SharedMemoryChannel(name, is_writer=False, timeout=30.0)
            shm_channels.append((name, ch))
            logger.info("[Inference Process] Connected to SHM: {}", name)
        except Exception as e:
            logger.warning("[Inference Process] Failed to connect to {}: {}", name, e)
    
    if not shm_channels:
        logger.error("[Inference Process] No SHM channels connected. Exiting.")
        return
    
    # Create synchronizer
    shm_synchronizer = SharedMemoryDataSynchronizer(
        shm_channels=shm_channels,
        buffer_maxlen=100,
        max_tolerance_s=0.05,
    ) 
    
    # Create chunk_shm for outputting action chunks
    chunk_shm = SharedMemoryChannel(name='chunk_shm', max_size_mb=10, is_writer=True)
    
    # Get obs2meta function
    # Try to import the robot module and get obs2meta
    obs2meta_func = None
    if robot_module_name:
        try:
            robot_module = importlib.import_module(robot_module_name)
            if hasattr(robot_module, 'obs2meta'):
                obs2meta_func = robot_module.obs2meta
        except Exception as e:
            logger.warning("[Inference Process] Could not load obs2meta from module: {}", e)
    

    
    if obs2meta_func is None:
        from deploy.base import default_obs2meta
        obs2meta_func = default_obs2meta
        logger.info("[Inference Process] Using default obs2meta function")
    
    logger.info("[Inference Process] Starting inference loop...")
    rate_limiter = RateLimiter()
    
    with torch.no_grad():
        try:
            while True:
                # Read and buffer data
                shm_synchronizer.read_and_buffer()
                
                # Get synced frame
                synced_data = shm_synchronizer.get_synced_frame(time.perf_counter())
                if synced_data is None:
                    time.sleep(0.001)
                    continue
                
                # Convert to policy observation format
                try:
                    policy_obs = obs2meta_func(synced_data)
                except Exception as e:
                    policy_obs = default_obs2meta(synced_data)
                
                # Run policy inference
                action = policy.select_action(policy_obs)
                
                # Write action chunk to SHM
                if action is not None:
                    action_data = {
                        'action': action if isinstance(action, np.ndarray) else np.array(action),
                        'timestamp': time.perf_counter(),
                    }
                    chunk_shm.write(action_data)
                
                rate_limiter.sleep(args.infer_rate)
                
        except KeyboardInterrupt:
            logger.info("[Inference Process] Exit by KeyboardInterrupt")
        except Exception as e:
            logger.error("[Inference Process] Error: {}", e)
            traceback.print_exc()
        finally:
            # Cleanup
            for _, ch in shm_channels:
                try:
                    ch.destroy()
                except Exception:
                    pass
            try:
                chunk_shm.destroy()
            except Exception:
                pass


def main():
    set_seed(0)
    args = parse_param()
    args.is_training = False
    
    # Load device configs
    robot_configs, all_shm_names = load_device_configs(args, getattr(args, 'unknown_args', []))
    
    # Get robot module info for obs2meta function in inference process
    robot_module_name, robot_class_name = get_robot_module_info(robot_configs)
    
    # Clean orphaned SHM
    shm_to_clean = all_shm_names + ['policy_control_shm', 'chunk_shm']
    cleanup_all_shm(shm_to_clean)
    logger.info("Cleaned up orphaned SHM segments.")
    
    # Create policy control SHM (main process writes actions here)
    control_shm = SharedMemoryChannel(name='policy_control_shm', max_size_mb=1, is_writer=True)
    logger.info("Created policy_control_shm for action publishing.")
    
    # Start all devices in subprocesses
    logger.info("Starting all devices...")
    device_procs = start_devices(robot_configs)
    time.sleep(2.0)
    
    # Start visualizer subprocesses if requested
    viz_procs = []
    if args.visualize:
        viz_procs = start_all_visualizers(
            device_configs=robot_configs,
            get_shm_name_func=get_shm_name,
            is_camera_config_func=is_camera_config,
        )
    
    # Start inference process
    logger.info("Starting inference process...")
    inference_proc = mp.Process(
        target=inference_process,
        args=(args, all_shm_names, robot_module_name, robot_class_name),
        daemon=True
    )
    inference_proc.start()
    time.sleep(1.0)
    
    # Connect to chunk_shm (read action chunks from inference process)
    try:
        chunk_shm = SharedMemoryChannel(name='chunk_shm', is_writer=False, timeout=30.0)
        logger.info("Connected to chunk_shm for reading action chunks.")
    except Exception as e:
        logger.error("Failed to connect to chunk_shm: {}", e)
        # Cleanup and exit
        inference_proc.terminate()
        for p in device_procs:
            p.terminate()
        control_shm.destroy()
        return
    
    # Load action manager configuration
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    logger.info("Loading action manager: {}", args.action_manager)
    try:
        manager_cfg, manager_cfg_path = cfg_loader.load_manager(args.action_manager)
        logger.info("✓ Loaded config from: {}", manager_cfg_path)
        logger.info("  Manager: {}", manager_cfg.get('manager_name', manager_cfg.get('name')))
        params_str = ', '.join(f'{k}={v}' for k, v in manager_cfg.items() 
                               if k not in ['name', 'manager_name', 'module_path', 'class_name'])
        if params_str:
            logger.info("  Parameters: {}", params_str)
    except Exception as e:
        # Fallback for legacy class names
        logger.info("Using legacy loading for: {}", args.action_manager)
        manager_cfg = {'manager_name': args.action_manager}
    
    action_manager = load_action_manager(config=manager_cfg)
    
    # Main control loop
    logger.info("="*60)
    logger.info("[Main Control Loop] Started. Press Ctrl+C to stop.")
    logger.info("  Publish rate: {} Hz", args.publish_rate)
    logger.info("  Inference rate: {} Hz", args.infer_rate)
    logger.info("="*60)
    
    rate_limiter = RateLimiter()
    action_count = 0
    last_log_time = time.perf_counter()
    
    try:
        while True:
            t = time.perf_counter()
            
            # Read action chunk from inference process
            action_chunk = chunk_shm.read(skip_unchanged=True, blocking=False)
            if action_chunk is not None:
                # Put new chunk into action manager
                chunk_action = action_chunk.get('action')
                if chunk_action is not None:
                    action_manager.put(chunk_action)
            
            # Get next action from action manager
            action = action_manager.get(t)
            
            # Write action to control_shm for robot to read
            if action is not None:
                action_data = {
                    'action': action if isinstance(action, np.ndarray) else np.array(action),
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
        logger.info("Cleaning up...")
        
        # Stop visualizers
        stop_all_visualizers(viz_procs)
        
        # Stop inference process
        if inference_proc.is_alive():
            inference_proc.terminate()
            inference_proc.join(timeout=2.0)
            if inference_proc.is_alive():
                inference_proc.kill()
        
        # Stop device processes
        for p in device_procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
                if p.is_alive():
                    p.kill()
        
        # Cleanup SHM
        try:
            control_shm.destroy()
        except Exception:
            pass
        try:
            chunk_shm.destroy()
        except Exception:
            pass
        
        cleanup_all_shm(shm_to_clean)
        logger.info("Done.")


if __name__ == '__main__':
    main()
