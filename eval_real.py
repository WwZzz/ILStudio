# eval_real.py (argparse version)
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
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
from configuration.constants import TASK_CONFIGS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
local_rank = None

action_lock = threading.Lock()

@dataclass
class HyperArguments:
    robot_config: str = "configuration/robots/dummy.yaml"
    # publish_rate决定动作消费的频率，如果太慢，会导致动作还没消费就被顶替（缓冲区较小时），造成大的抖动；如果消费太快，动作生产者跟不上消费的速度，会造成很多卡顿；此外，机器执行跟不上消费的速度，也会导致动作被发送却没执行完毕，造成动作被浪费；
    publish_rate: int = 25
    # sensing_rate和freq共同决定推理的频率，一方面是需要推理时需要观测，此时没观测会被阻塞，所以观测率不应该太低；另一方面freq决定了多少步推理一次，上回推理结果没用完不会执行推理；
    sensing_rate: int = 200
    freq: int = 100 # 对应每次推理往动作缓冲区推送的总动作数量；
    # chunk_size决定了动作缓冲区的大小，太大则会导致存储了很多过时的动作，若消费不及时，则导致动作跟不上观测，执行慢半拍；太小的话若是消费不及时，容易造成大的跳跃式动作；这个越小，消费应越快；
    chunk_size: int = field(default=10)
    action_buffer_size: int = field(default=10)

    # ############## model  ################
    is_pretrained: bool = field(default=True)
    device: str = 'cuda'
    model_name: str = 'act'
    model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/act_sfov1_insertion_top_zscore_tau_0.01"
    norm_path: str = ''
    save_dir: str = 'results/real_debug'
    dataset_dir: str = '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_insertion_scripted'
    task: str = field(default="sim_transfer_cube_scripted")

    fps: int = 50
    num_rollout: int = 4
    max_timesteps: int = 400
    image_size: str = '(640, 480)'  # (width, height)
    ctrl_space: str = 'joint'
    ctrl_type: str = 'abs'
    camera_ids: str = '[0]'
    #  ############ data ###################
    image_size_primary: str = "(640,480)"  # image size of non-wrist camera
    image_size_wrist: str = "(640,480)"  # image size of wrist camera
    camera_names: List[str] = field(
        default_factory=lambda: ['primary'],
        metadata={"help": "List of camera names", "nargs": "+"}
    )


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def parse_param():
    global local_rank
    # Use HFParser to pass parameters, which are defined in the dataclass above
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    print(unknown_args)
    print(args)
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]  # Remove '--' prefix
            if "=" in key:
                key, value = key.split('=', 1)
            else:
                value = True  # If no value is specified (e.g., --flag), default to True
            extra_args[key] = value
    model_args = {}
    # Dynamically inject `extra_args` into the args object
    for key, value in extra_args.items():
        try:
            value = _convert_to_type(value)
            if key.startswith('model.'):
                model_args[key[6:]] = value  # Dynamically get custom model_args, i.e., strings starting with model.
            else:
                setattr(args, key, value)  # Set non-model-related parameters as attributes of args
        except ValueError as e:
            print(f"Warning: {e}")
    args.model_args = model_args
    return args

# from deploy.robots.ros_robot import ROSRobot # Example for a real robot

def make_robot(robot_cfg: Dict, args):
    """
    Factory function to create a robot instance from a config dictionary.

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

    robot = RobotCls(config=robot_config, extra_args=args)
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
                    # Non-blocking: Put data into the queue
                    if not observation_queue.full():
                        observation_queue.put((obs, t_obs))
                    else:
                        try:
                            observation_queue.get_nowait()
                        except queue.Empty:
                            pass
                        observation_queue.put((obs, t_obs))
            # else:
            #     print("[Sensing Thread] No Observation Found...")
            rate_limiter.sleep(args.sensing_rate)
    except Exception as e:
        print(f"[Sensing Thread] An exception occurred: {e}")
        traceback.print_exc()
        robot.shutdown()


def inference_producer(policy, observation_queue: queue.Queue, action_queue: queue.Queue, args):
    """Inference producer thread, consumes observation data and produces actions."""
    print("[Inference Thread] Producer started.")
    global action_lock
    with torch.no_grad():
        try:
            t = 0
            while True:
                # Blocking: Wait for observation data
                obs, t_obs = observation_queue.get()
                obs.to_batch()
                # Blocking: Execute model inference
                act = policy.select_action(obs, t, return_all=False)
                t += 1
                action_lock.acquire()
                for i in range(len(act)):
                    if action_queue.full():
                        try:
                            action_queue.get_nowait()
                        except queue.Empty:
                            pass
                    action_queue.put(act[i])
                action_lock.release()
        except Exception as e:
            print(f"[Inference Thread] An exception occurred: {e}")
            traceback.print_exc()
            robot.shutdown()


if __name__ == '__main__':
    set_seed(0)
    args = parse_param()
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
    # --- 1. Load Policy ---
    model_module = importlib.import_module(f"vla.{args.model_name}")
    assert hasattr(model_module,
                   'load_model'), "model_name must provide API named `load_model` that returns dict like '\{'model':...\}'"
    model_components = model_module.load_model(args)  # load_model is an interface that the model module must implement
    model = model_components['model'].to('cuda')
    policy = MetaPolicy(policy=model, freq=args.freq, action_normalizer=normalizers['action'],
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
    action_queue = queue.Queue(maxsize=args.chunk_size)

    # Start producer and consumer threads
    sensing_thread = threading.Thread(target=sensing_producer, args=(robot, observation_queue, args))
    inference_thread = threading.Thread(target=inference_producer,
                                        args=(policy, observation_queue, action_queue, args))

    sensing_thread.daemon = True
    inference_thread.daemon = True

    sensing_thread.start()
    inference_thread.start()

    print("[Main Control Loop] Consumer started.")
    try:
        rate_limiter = RateLimiter()
        while robot.is_running():
            if not action_queue.empty():
                action_lock.acquire()
                action = action_queue.get()
                action_lock.release()
                action = robot.meta2act(action)
                print(f"[Main Control Loop] New action {action} found, updating...")
                robot.publish_action(action)
            rate_limiter.sleep(args.publish_rate)
    except KeyboardInterrupt:
        print(f"[Main Control Loop] An exception occurred: {e}")
        robot.shutdown()