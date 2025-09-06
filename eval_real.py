# eval_real.py (simplified argparse with complete core functionality)
import yaml
import os
import traceback
import time
import transformers
import importlib
import threading
import queue
import torch
import numpy as np
from benchmark.base import MetaPolicy
from data_utils.utils import set_seed,  _convert_to_type, load_normalizers
from deploy.robot.base import AbstractRobotInterface, RateLimiter
from PIL import Image, ImageDraw, ImageFont
from configs.task.loader import load_task_config
from typing import Dict, Optional, Sequence, List, Any
from deploy.action_manager import load_action_manager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
local_rank = None

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model on real robot')
    
    # Robot configuration
    parser.add_argument('--robot_config', type=str, default='configs/robots/dummy.yaml',
                       help='Robot configuration file')
    parser.add_argument('--publish_rate', type=int, default=25,
                       help='Action publishing rate (Hz)')
    parser.add_argument('--sensing_rate', type=int, default=20,
                       help='Sensing rate (Hz)')
    
    # Model arguments
    parser.add_argument('--is_pretrained', action='store_true', default=True,
                       help='Whether to use pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    # Policy config system
    parser.add_argument('--policy_config', type=str, default='configs/policy/act.yaml',
                       help='Policy config file path')
    parser.add_argument('--model_name_or_path', type=str, 
                       default='/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/act_sfov1_insertion_top_zscore_tau_0.01',
                       help='Path to the model checkpoint')
    parser.add_argument('--norm_path', type=str, default='',
                       help='Path to normalization data')
    parser.add_argument('--save_dir', type=str, default='results/real_debug',
                       help='Directory to save results')
    parser.add_argument('--dataset_dir', type=str, 
                       default='/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_insertion_scripted',
                       help='Dataset directory')
    parser.add_argument('--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task name')
    
    # Evaluation parameters
    parser.add_argument('--fps', type=int, default=50,
                       help='Frames per second')
    parser.add_argument('--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('--max_timesteps', type=int, default=400,
                       help='Maximum timesteps per episode')
    parser.add_argument('--image_size', type=str, default='(640, 480)',
                       help='Image size (width, height)')
    parser.add_argument('--ctrl_space', type=str, default='joint',
                       help='Control space')
    parser.add_argument('--ctrl_type', type=str, default='abs',
                       help='Control type')
    parser.add_argument('--camera_ids', type=str, default='[0]',
                       help='Camera IDs')
    
    # Action manager
    parser.add_argument('--action_manager', type=str, default='OlderFirstManager',
                       help='Action manager type')
    parser.add_argument('--manager_coef', type=float, default=1.0,
                       help='Action manager coefficient')
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def make_robot(robot_cfg: Dict, args) -> AbstractRobotInterface:
    """
    Create a robot instance based on the provided configuration.

    Args:
        robot_cfg (Dict): A dictionary loaded from the robot's YAML config file.
    """
    full_path = robot_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    RobotCls = getattr(module, class_name)
    print(f"Creating robot: {full_path}")

    # .get() is used to safely access params, which might not exist
    robot_config = robot_cfg.get('config', {})

    robot = RobotCls(config=robot_config, extra_args=args, **robot_config)
    # connect to robot
    retry_counts = 1
    MAX_RETRY = 5
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts>MAX_RETRY:
            exit(0)
        time.sleep(1)
    return robot

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
    
    # For evaluation, parameters will be loaded from saved model config
    # No need to load task config parameters
    
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
    
    # --- 1. Load Policy ---
    # Use policy config system for evaluation - uses saved model config
    print(f"Loading policy config: {args.policy_config}")
    from policy.policy_loader import load_policy_model_for_evaluation
    model_components = load_policy_model_for_evaluation(args.policy_config, args)
    model = model_components['model'].to('cuda')
    config = model_components.get('config', None)
    if config:
        print(f"Loaded config from YAML: {type(config).__name__}")
    policy = MetaPolicy(policy=model, chunk_size=args.chunk_size, action_normalizer=normalizers['action'],
                        state_normalizer=normalizers['state'], ctrl_space=ctrl_space, ctrl_type=ctrl_type)

    # --- 2. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)

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