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
import traceback
import time
import numpy as np
from loguru import logger

from data_utils.utils import set_seed
from deploy.base import is_camera_config
from deploy.shm_utils import SharedMemoryChannel, cleanup_all_shm
from deploy.utils import (
    RateLimiter,
    get_shm_name,
    load_device_configs,
    start_devices,
)
from deploy.action_manager import load_action_manager
from deploy.inference import start_inference_process, stop_inference_process
from deploy.visualizer.base import start_all_visualizers
from configs.loader import ConfigLoader


def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model on real robot')
    
    # Same primary flags as collect_data.py (-r / --robot); long --robot_config kept as alias.
    parser.add_argument(
        '-r', '--robot', '--robot-config', '--robot_config',
        type=str,
        default='robot/so101_follower',
        help="Robot config (name under configs/robot or path to yaml; same as collect_data.py -r/--robot)",
    )
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
    
    # Same loader as collect_data; eval forces policy_control_shm on robots.
    robot_configs, all_shm_names, _teleop_configs, visualizer_configs = load_device_configs(
        args,
        getattr(args, "unknown_args", []),
        cfg_loader=cfg_loader,
        force_robot_control_shm="policy_control_shm",
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
