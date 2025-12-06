import configs 
import os
import yaml
import traceback
import time
import threading
import queue
import imageio
import torch
from loguru import logger
from data_utils.utils import set_seed,  _convert_to_type
from deploy.robot.base import AbstractRobotInterface, RateLimiter, make_robot
from deploy.action_manager import load_action_manager
from policy.utils import load_policy
import numpy as np

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model on real robot')
    
    # Robot configuration
    parser.add_argument('-c', '--robot_config', type=str, default='robot/dummy',
                       help='Robot config (name under configs/robot or absolute path to yaml)')
    parser.add_argument('-pr', '--publish_rate', type=float, default=25,
                       help='Action publishing rate (Hz)')
    parser.add_argument('-sr', '--sensing_rate', type=float, default=20,
                       help='Sensing rate (Hz)')
    
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
                       help='Action manager config name or path to config file (e.g., basic, older_first, temporal_agg, configs/action_manager/custom.yaml)')
    
    # Parse arguments (allow unknown for dotted overrides like --robot.xxx, --manager.xxx)
    args, unknown = parser.parse_known_args()
    
    # Store unknown args for ConfigLoader to process
    setattr(args, 'unknown_args', unknown)
    return args


def sensing_producer(robot: AbstractRobotInterface, observation_queue: queue.Queue, args):
    """Sensing producer thread, uses an abstract interface to get observations."""
    logger.info("[Sensing Thread] Producer started.")
    if args.output_dir!='':
        os.makedirs(args.output_dir, exist_ok=True)
        if args.episode_id<0:
            records = os.listdir(args.output_dir)
            ids = [int(r.split('video')[-1].split('.mp4')[0]) for r in records if r.endswith('.mp4')]
            args.episode_id = max(ids)+1 if len(ids)>0 else 0
        video_path = os.path.join(args.output_dir, f"video{args.episode_id:03d}.mp4")
        video_writer = imageio.get_writer(video_path, fps=args.sensing_rate, codec='libx264')
    else:
        video_writer = None
    try:
        rate_limiter = RateLimiter()
        while robot.is_running():
            # Blocking: Call interface to get synchronous data
            obs = robot.get_observation()
            t_obs = time.perf_counter()
            if obs:
                logger.debug(f"[Sensing Thread] New Observation came at {args.sensing_rate}Hz...")
                obs = robot.obs2meta(obs)
                if observation_queue.full():
                    try:
                        observation_queue.get_nowait()
                    except queue.Empty:
                        pass
                # Non-blocking: Put data into the queue
                observation_queue.put((obs, t_obs))
                if video_writer is not None:
                # --- Save image to video ---
                    img = obs['image'][0]
                    # img: np.ndarray, shape (C, H, W) or (H, W, C)
                    if img.ndim == 3 and img.shape[0] in [1, 3]:
                        img = np.transpose(img, (1, 2, 0))
                    img = np.ascontiguousarray(img)
                    # Convert float to uint8 if needed
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    video_writer.append_data(img)
            rate_limiter.sleep(args.sensing_rate)
    except Exception as e:
        logger.error(f"[Sensing Thread] An exception occurred: {e}")
        traceback.print_exc()
        robot.shutdown()

def inference_producer(policy, observation_queue: queue.Queue, action_manager: queue.Queue, args):
    """Inference producer thread, consumes observation data and produces actions."""
    logger.info("[Inference Thread] Producer started.")
    with torch.no_grad():
        try:
            step_count = 0
            while True:
                # Blocking: Wait for observation data
                obs, t_obs = observation_queue.get()
                obs.to_batch()
                # Blocking: Execute model inference
                raw_action_chunk = policy.inference(obs)
                action_chunk = [aci[0] for aci in raw_action_chunk]
                step_count += 1
                action_manager.put(action_chunk, t_obs)
        except Exception as e:
            logger.error(f"[Inference Thread] An exception occurred: {e}")
            traceback.print_exc()
            robot.shutdown()

if __name__ == '__main__':
    set_seed(0)
    args = parse_param()
    args.is_training = False
    
    # Build config loader from CLI for overrides
    # Explicitly pass unknown_args to ConfigLoader (same as train.py)
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    
    # For evaluation, parameters will be loaded from saved model config
    # No need to load task config parameters
    policy = load_policy(args)
    
    # check policy
    if not hasattr(args, 'ctrl_space'):
        args.ctrl_space = policy.ctrl_space
        args.ctrl_type = policy.ctrl_type

    # --- 2. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    from configs.utils import apply_overrides_to_mapping
    from data_utils.utils import _convert_to_type
    # parse unknown overrides here as well
    # use overrides parsed by ConfigLoader
    try:
        robot_cfg_path = cfg_loader._resolve('robot', args.robot_config)
    except Exception:
        robot_cfg_path = args.robot_config
    logger.info(f"Loading robot configuration from {robot_cfg_path}")
    with open(robot_cfg_path, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    apply_overrides_to_mapping(robot_cfg, cfg_loader.get_overrides('robot'), _convert_to_type)

    robot = make_robot(robot_cfg, args)
    
    logger.info("Robot successfully loaded.")
    input("=" * 10 + "Press Enter to start evaluation..." + "=" * 10)

    # Create thread-safe queues
    observation_queue = queue.Queue(maxsize=1)

    # Load action manager configuration using ConfigLoader
    # Supports:
    # - Config names: 'truncated_conservative', 'older_first', etc.
    # - Config file paths: 'configs/action_manager/my_custom.yaml'
    # - Command-line overrides: --manager.start_ratio 0.15 --manager.end_ratio 0.25
    # - Dynamic class loading via module_path and class_name in config
    logger.info(f"Loading action manager: {args.action_manager}")
    try:
        manager_cfg, manager_cfg_path = cfg_loader.load_manager(args.action_manager)
        logger.info(f"✓ Loaded config from: {manager_cfg_path}")
        logger.info(f"  Manager: {manager_cfg.get('manager_name', manager_cfg.get('name'))}")
        logger.info(f"  Parameters: {', '.join(f'{k}={v}' for k, v in manager_cfg.items() if k not in ['name', 'manager_name', 'module_path', 'class_name'])}")
    except Exception as e:
        # Fallback for legacy class names
        logger.info(f"Using legacy loading for: {args.action_manager}")
        manager_cfg = {'manager_name': args.action_manager}
    
    action_manager = load_action_manager(config=manager_cfg)

    # Start producer and consumer threads
    sensing_thread = threading.Thread(target=sensing_producer, args=(robot, observation_queue, args))
    inference_thread = threading.Thread(target=inference_producer,
                                        args=(policy, observation_queue, action_manager, args))

    sensing_thread.daemon = True
    inference_thread.daemon = True

    sensing_thread.start()
    inference_thread.start()

    logger.info("[Main Control Loop] Consumer started.")
    try:
        rate_limiter = RateLimiter()
        while robot.is_running():
            # if not action_manager.empty():
            t = time.perf_counter()
            action = action_manager.get(t)
            if action is not None:
                action = robot.meta2act(action)
                logger.debug(f"[Main Control Loop] New action {action} found, updating...")
                robot.publish_action(action)
            rate_limiter.sleep(args.publish_rate)
    except KeyboardInterrupt:
        logger.info(f"[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
        robot.shutdown()