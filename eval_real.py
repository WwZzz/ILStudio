# eval_real.py (simplified argparse with complete core functionality)
import yaml
import traceback
import time
import threading
import queue
import torch
from data_utils.utils import set_seed,  _convert_to_type
from deploy.robot.base import AbstractRobotInterface, RateLimiter, make_robot
from deploy.action_manager import load_action_manager
from policy.utils import load_policy

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
    parser.add_argument('-pr', '--publish_rate', type=int, default=25,
                       help='Action publishing rate (Hz)')
    parser.add_argument('-sr', '--sensing_rate', type=int, default=20,
                       help='Sensing rate (Hz)')
    
    # Model arguments
    parser.add_argument('--is_pretrained', action='store_true', default=True,
                       help='Whether to use pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    # Direct checkpoint loading
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='localhost:5000',
                       help='Path to model checkpoint OR server address (host:port) for remote policy server')
    parser.add_argument('--norm_path', type=str, default='',
                       help='Path to normalization data')
    parser.add_argument('--save_dir', type=str, default='results/real_debug',
                       help='Directory to save results')
    parser.add_argument('--dataset_id', type=str, default='',
                       help='Dataset ID to use (if multiple datasets, defaults to first)')
    parser.add_argument('--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task config (name under configs/task or absolute path to yaml)')
    
    # Evaluation parameters
    parser.add_argument('--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('--max_timesteps', type=int, default=400,
                       help='Maximum timesteps per episode')
    parser.add_argument('--image_size', type=str, default='(640, 480)',
                       help='Image size (width, height)')
    parser.add_argument('--camera_ids', type=str, default='[0]',
                       help='Camera IDs')
    
    # Action manager
    parser.add_argument('--action_manager', type=str, default='OlderFirstManager',
                       help='Action manager type')
    parser.add_argument('--manager_coef', type=float, default=1.0,
                       help='Action manager coefficient')
    
    # Parse arguments
    args, unknown = parser.parse_known_args()
    return args


def sensing_producer(robot: AbstractRobotInterface, observation_queue: queue.Queue, args):
    """Sensing producer thread, uses an abstract interface to get observations."""
    print("[Sensing Thread] Producer started.")
    try:
        rate_limiter = RateLimiter()
        while robot.is_running():
            # Blocking: Call interface to get synchronous data
            obs = robot.get_observation()
            t_obs = time.perf_counter()
            if obs:
                print(f"[Sensing Thread] New Observation came at {args.sensing_rate}Hz...")
                obs = robot.obs2meta(obs)
                if obs:
                    if observation_queue.full():
                        try:
                            observation_queue.get_nowait()
                        except queue.Empty:
                            pass
                    # Non-blocking: Put data into the queue
                    observation_queue.put((obs, t_obs))
            rate_limiter.sleep(args.sensing_rate)
    except Exception as e:
        print(f"[Sensing Thread] An exception occurred: {e}")
        traceback.print_exc()
        robot.shutdown()

def inference_producer(policy, observation_queue: queue.Queue, action_manager: queue.Queue, args):
    """Inference producer thread, consumes observation data and produces actions."""
    print("[Inference Thread] Producer started.")
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
            print(f"[Inference Thread] An exception occurred: {e}")
            traceback.print_exc()
            robot.shutdown()

if __name__ == '__main__':
    set_seed(0)
    args = parse_param()
    # Build config loader from CLI for overrides
    import sys
    unknown = [tok for tok in sys.argv[1:] if tok.startswith('--robot.')]
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
    
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
    print(f"Loading robot configuration from {robot_cfg_path}")
    with open(robot_cfg_path, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    apply_overrides_to_mapping(robot_cfg, cfg_loader.get_overrides('robot'), _convert_to_type)

    robot = make_robot(robot_cfg, args)
    
    print("Robot successfully loaded.")
    input("=" * 10 + "Press Enter to start evaluation..." + "=" * 10)

    # Create thread-safe queues
    observation_queue = queue.Queue(maxsize=1)

    # init action manager
    action_manager = load_action_manager(args.action_manager, args)

    # Start producer and consumer threads
    sensing_thread = threading.Thread(target=sensing_producer, args=(robot, observation_queue, args))
    inference_thread = threading.Thread(target=inference_producer,
                                        args=(policy, observation_queue, action_manager, args))

    sensing_thread.daemon = True
    inference_thread.daemon = True

    sensing_thread.start()
    inference_thread.start()

    print("[Main Control Loop] Consumer started.")
    try:
        rate_limiter = RateLimiter()
        while robot.is_running():
            # if not action_manager.empty():
            t = time.perf_counter()
            action = action_manager.get(t)
            if action is not None:
                action = robot.meta2act(action)
                print(f"[Main Control Loop] New action {action} found, updating...")
                robot.publish_action(action)
            rate_limiter.sleep(args.publish_rate)
    except KeyboardInterrupt:
        print(f"[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
        robot.shutdown()