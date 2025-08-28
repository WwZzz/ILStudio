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
from data_utils.utils import set_seed, _convert_to_type, load_normalizers
from deploy.robot.base import AbstractRobotInterface, RateLimiter
from deploy.teleoperator.base import str2dtype
from PIL import Image, ImageDraw, ImageFont
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
from configuration.constants import TASK_CONFIGS
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod


# action_lock = threading.Lock()

@dataclass
class HyperArguments:
    shm_name: str = 'teleop_action_buffer'
    action_dtype: str = 'float64'
    action_dim: int = 7
    robot_config: str = "configuration/robots/dummy.yaml"
    # publish_rate决定动作消费的频率，如果太慢，会导致动作还没消费就被顶替（缓冲区较小时），造成大的抖动；如果消费太快，动作生产者跟不上消费的速度，会造成很多卡顿；此外，机器执行跟不上消费的速度，也会导致动作被发送却没执行完毕，造成动作被浪费；
    save_dir: str = 'data/debug'
    task: str = field(default="sim_transfer_cube_scripted")

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


def make_robot(robot_cfg: Dict, args, max_connect_retry: int=5):
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

    robot = RobotCls( **robot_config, extra_args=args)
    # connect to robot
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot


class RobotController:
    """
    从共享内存读取动作并发送给机器人
    """

    def __init__(self, robot, shm_name, shm_shape, shm_dtype):
        self.robot = robot
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.shm = None
        self.action_buffer = None
        self.last_timestamp = 0.0
        self.stop_event = mp.Event()

    def connect_to_buffer(self):
        """连接到共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("机器人控制器：成功连接到共享内存。")
        except FileNotFoundError:
            print(f"机器人控制器：错误！共享内存 '{self.shm_name}' 不存在。")
            raise

    def run(self):
        """
        主循环，非阻塞地从缓冲区获取数据并发送给机器人
        """
        self.connect_to_buffer()
        try:
            while not self.stop_event.is_set():
                current_timestamp = self.action_buffer[0]['timestamp']
                if current_timestamp > self.last_timestamp:
                    self.last_timestamp = current_timestamp
                    action = self.action_buffer[0]['action'].copy()
                    self.robot.publish_action(action)
        finally:
            print("\n机器人控制器：正在关闭...")
            if self.shm:
                self.shm.close()

    def stop(self):
        self.stop_event.set()

if __name__ == '__main__':
    set_seed(0)
    args = parse_param()

    # --- 1. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    robot = make_robot(robot_cfg, args)
    print("Robot successfully loaded.")


    # --- 2. Connect to shm
    args.action_dtype = str2dtype(args.action_dtype)
    shm_info = {
        'name': args.shm_name, 
        'dtype': np.dtype([
            ('timestamp', np.float64),
            ('action', args.action_dtype, args.action_dim), 
        ]),
        'shape': (1,),
    }
    shm_info['size'] = shm_info['dtype'].itemsize
    robot_controller = RobotController(
        robot = robot,
        shm_name=shm_info['name'],
        shm_shape=shm_info['shape'],
        shm_dtype=shm_info['dtype'],
    )

    input("=" * 10 + "Press Enter to collect data..." + "=" * 10)

    # Create thread-safe queues

    print("[Main Control Loop] Consumer started.")
    try:
        robot_controller.run()
    except KeyboardInterrupt:
        print(f"[Main Control Loop] Exit by KeyboardInterrupt Ctrl+C")
        robot.shutdown()

