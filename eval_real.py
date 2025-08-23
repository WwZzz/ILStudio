# eval_real.py (argparse version)
import yaml
import os
import sys
import traceback
import time
import copy
import json
import argparse
import imageio
import transformers
import importlib
import IPython
import threading
import math
import queue
import torch
import numpy as np
from benchmark.base import MetaPolicy
from data_utils.utils import set_seed, load_normalizer_from_meta
from deploy.robot.base import AbstractRobotInterface
from PIL import Image, ImageDraw, ImageFont
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
from configuration.constants import TASK_CONFIGS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
e = IPython.embed
local_rank = None


@dataclass
class HyperArguments:
    robot_config: str = "configuration/robots/dummy.yaml"
    publish_rate: int = 20
    sensing_rate: int = 50
    # ############## model  ################
    is_pretrained: bool = field(default=True)
    device: str = 'cuda'
    model_name: str = 'act'
    model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/act_sfov1_insertion_top_zscore_tau_0.01"
    norm_path: str = ''
    chunk_size: int = field(default=50)
    freq: int = 50
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

def _convert_to_type(value):
    """
    Infers the type of a value based on its format. Supports int, float, and bool.
    """
    if not isinstance(value, str): return value
    # Attempt to infer boolean value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    # Attempt to infer integer type
    if value.isdigit():
        return int(value)
    # Attempt to infer float type
    try:
        return float(value)
    except ValueError:
        pass
    # Otherwise, return the original string
    return value


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

def load_normalizers(args):
    # load normalizers
    if args.norm_path == '':
        res = os.path.join(os.path.dirname(args.model_name_or_path), 'normalize.json')
        if not os.path.exists(res):
            res = os.path.join(args.model_name_or_path, 'normalize.json')
            if not os.path.exists(res):
                raise FileNotFoundError("No normalize.json found")
    elif args.norm_path=='identity':
        from data_utils.normalize import Identity
        normalizers = dict(state=Identity(ctrl_type=args.ctrl_type, ctrl_space=args.ctrl_space), action=Identity(ctrl_type=args.ctrl_type, ctrl_space=args.ctrl_space))
        return normalizers, args.ctrl_space, acts.ctrl_type
    else:
        res = args.norm_path
    with open(res, 'r') as f:
        norm_meta = json.load(f)
    normalizers = load_normalizer_from_meta(args.dataset_dir, norm_meta)
    kwargs = norm_meta.get('kwargs', {'ctrl_type': 'delta', 'ctrl_space': 'ee'})
    return normalizers, kwargs['ctrl_space'], kwargs['ctrl_type']


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
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        time.sleep(1)
    return robot


def sensing_producer(robot: AbstractRobotInterface, observation_queue: queue.Queue, args):
    """Sensing producer thread, uses an abstract interface to get observations."""
    print("[Sensing Thread] Producer started.")
    try:
        while robot.is_running():
            # Blocking: Call interface to get synchronous data
            obs = robot.get_observation()
            obs = robot.obs2meta(obs)
            if obs:
                # Non-blocking: Put data into the queue
                if not observation_queue.full():
                    observation_queue.put(obs)
                else:
                    try:
                        observation_queue.get_nowait()
                    except queue.Empty:
                        pass
                    observation_queue.put(obs)
            robot.rate_sleep(args.sensing_rate)
    except Exception as e:
        print(f"[Sensing Thread] An exception occurred: {e}")


def inference_producer(policy, normalizers, observation_queue: queue.Queue, action_queue: queue.Queue, args):
    """Inference producer thread, consumes observation data and produces actions."""
    print("[Inference Thread] Producer started.")
    with torch.no_grad():
        try:
            t = 0
            while True:
                # Blocking: Wait for observation data
                obs = observation_queue.get()
                obs.to_batch()
                # Blocking: Execute model inference
                act = policy.select_action(obs, t)[0]
                t += 1
                # Non-blocking: Put action into the queue
                if not action_queue.full():
                    action_queue.put(act)
                else:
                    try:
                        action_queue.get_nowait()
                    except queue.Empty:
                        pass
                    action_queue.put(act)

        except Exception as e:
            print(f"[Inference Thread] An exception occurred: {e}")
            traceback.print_exc()


def action_consumer(robot: AbstractRobotInterface, args, action_queue: queue.Queue):
    """Main control loop, consumes actions and publishes them."""
    print("[Main Control Loop] Consumer started.")
    try:
        while robot.is_running():
            if not action_queue.empty():
                print("[Main Control Loop] New action found, updating...")
                action = action_queue.get()
                action = robot.meta2act(action)
                robot.publish_action(action)
            robot.rate_sleep(args.publish_rate)
    except Exception as e:
        print(f"[Main Control Loop] An exception occurred: {e}")
        traceback.print_exc()


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
    model = model_components['model']
    policy = MetaPolicy(policy=model, freq=args.freq, action_normalizer=normalizers['action'],
                        state_normalizer=normalizers['state'], ctrl_space=ctrl_space, ctrl_type=ctrl_type)

    # --- 2. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)

    robot = make_robot(robot_cfg, args)

    input("=" * 10 + "Press Enter to start evaluation..." + "=" * 10)

    # Create thread-safe queues
    observation_queue = queue.Queue(maxsize=1)
    action_queue = queue.Queue(maxsize=1)

    # Start producer and consumer threads
    sensing_thread = threading.Thread(target=sensing_producer, args=(robot, observation_queue, args))
    inference_thread = threading.Thread(target=inference_producer,
                                        args=(policy, normalizers, observation_queue, action_queue, args))
    control_thread = threading.Thread(target=action_consumer, args=(robot, args, action_queue))

    sensing_thread.daemon = True
    inference_thread.daemon = True
    control_thread.daemon = True

    sensing_thread.start()
    inference_thread.start()
    control_thread.start()

    print("\nAll threads started. Press Ctrl+C to exit.")
    try:
        while robot.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram terminated.")
        robot.shutdown_event.set()