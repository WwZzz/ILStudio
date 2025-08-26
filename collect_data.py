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
from PIL import Image, ImageDraw, ImageFont
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
from configuration.constants import TASK_CONFIGS
from deploy.action_manager import load_action_manager

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
local_rank = None


# action_lock = threading.Lock()

@dataclass
class HyperArguments:
    robot_config: str = "configuration/robots/dummy.yaml"
    # publish_rate决定动作消费的频率，如果太慢，会导致动作还没消费就被顶替（缓冲区较小时），造成大的抖动；如果消费太快，动作生产者跟不上消费的速度，会造成很多卡顿；此外，机器执行跟不上消费的速度，也会导致动作被发送却没执行完毕，造成动作被浪费；
    publish_rate: int = 25
    # sensing_rate和freq共同决定推理的频率，一方面是需要推理时需要观测，此时没观测会被阻塞，所以观测率不应该太低；另一方面freq决定了多少步推理一次，上回推理结果没用完不会执行推理；
    sensing_rate: int = 20
    chunk_size: int = 100  # 对应每次推理往动作缓冲区推送的总动作数量；

    # ############## model  ################
    is_pretrained: bool = field(default=True)
    device: str = 'cuda'
    save_dir: str = 'results/real_debug'
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

    robot = RobotCls(config=robot_config, extra_args=args)
    # connect to robot
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot

def load_teleop_dev(teleop_config, args):
    full_path = teleop_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    TeleOpCls = getattr(module, class_name)
    print(f"Creating Teleop Device: {full_path}")


if __name__ == '__main__':
    set_seed(0)
    args = parse_param()
    # --- 1. Load Teleop Input Device
    print(f"Loading robot configuration from {args.teleop_config}")


    # --- 2. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    robot = make_robot(robot_cfg, args)

    print("Robot successfully loaded.")
    input("=" * 10 + "Press Enter to collect data..." + "=" * 10)

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


import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard


# --- 1. 遥操作设备基类 ---

class TeleopDeviceBase(ABC):
    """
    遥操作设备抽象基类
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=100):
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.frequency = frequency
        self.shm = None
        self.action_buffer = None
        self.stop_event = mp.Event()

    def connect_to_buffer(self):
        """连接到已存在的共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("遥操作设备：成功连接到共享内存。")
        except FileNotFoundError:
            print(f"遥操作设备：错误！共享内存 '{self.shm_name}' 不存在。")
            raise

    @abstractmethod
    def get_observation(self):
        """从设备获取原始观测数据（例如，按键状态）"""
        pass

    @abstractmethod
    def observation_to_action(self, observation):
        """将观测数据转换为标准化的机器人动作"""
        pass

    def put_action_to_buffer(self, action):
        """将动作写入共享内存缓冲区"""
        if self.action_buffer is not None:
            self.action_buffer[:] = action

    def run(self):
        """
        主循环，以指定频率获取观测、转换动作并写入缓冲区
        """
        self.connect_to_buffer()
        rate = 1.0 / self.frequency
        try:
            while not self.stop_event.is_set():
                start_time = time.time()

                # 核心三步
                observation = self.get_observation()
                action = self.observation_to_action(observation)
                self.put_action_to_buffer(action)

                elapsed_time = time.time() - start_time
                sleep_time = rate - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("遥操作设备：正在关闭...")
            if self.shm:
                self.shm.close()

    def stop(self):
        """设置停止事件"""
        self.stop_event.set()


# --- 2. 键盘遥操作实现 ---

class KeyboardTeleop(TeleopDeviceBase):
    """
    使用键盘进行遥操作的具体实现
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=100):
        super().__init__(shm_name, shm_shape, shm_dtype, frequency)
        self.pressed_keys = set()

        # 将控制灵敏度定义为类内部的属性
        self.MIN_TRANS_STEP = 0.005  # 每次按键的最小平移量 (米)
        self.MIN_ROT_STEP = np.deg2rad(1.5)  # 每次按键的最小旋转量 (弧度)

        self._start_keyboard_listener()

    def _on_press(self, key):
        """按键按下时的回调函数"""
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            # 处理特殊按键，如方向键、空格等
            self.pressed_keys.add(key)

    def _on_release(self, key):
        """按键释放时的回调函数"""
        try:
            self.pressed_keys.remove(key.char)
        except AttributeError:
            self.pressed_keys.remove(key)
        except KeyError:
            pass  # 忽略已经不在集合中的按键

    def _start_keyboard_listener(self):
        """启动一个独立的线程来监听键盘事件"""
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True  # 设置为守护线程，主进程退出时它也退出
        listener.start()
        print("键盘监听器已启动。")

    def get_observation(self):
        """获取当前所有被按下的键"""
        return self.pressed_keys.copy()

    def observation_to_action(self, observation):
        """将按键状态转换为机械臂末端的 delta 位姿和夹爪动作"""
        action = np.zeros(self.shm_shape, dtype=self.shm_dtype)

        # 平移控制 (X, Y, Z)
        if 'a' in observation: action[0] = -self.MIN_TRANS_STEP
        if 'd' in observation: action[0] = self.MIN_TRANS_STEP
        if 'w' in observation: action[1] = self.MIN_TRANS_STEP
        if 's' in observation: action[1] = -self.MIN_TRANS_STEP
        if 'q' in observation: action[2] = self.MIN_TRANS_STEP
        if 'e' in observation: action[2] = -self.MIN_TRANS_STEP

        # 旋转控制 (Roll, Pitch, Yaw)
        if 'j' in observation: action[3] = self.MIN_ROT_STEP
        if 'l' in observation: action[3] = -self.MIN_ROT_STEP
        if 'i' in observation: action[4] = self.MIN_ROT_STEP
        if 'k' in observation: action[4] = -self.MIN_ROT_STEP
        if 'u' in observation: action[5] = self.MIN_ROT_STEP
        if 'o' in observation: action[5] = -self.MIN_ROT_STEP

        # 夹爪控制
        if keyboard.Key.space in observation:
            action[6] = -1.0  # 闭合信号
        else:
            action[6] = 1.0  # 张开信号

        return action


# --- 3. 机器人控制器 ---

class MockRobot:
    """一个模拟的机器人接口，用于演示"""

    def __init__(self):
        self.gripper_width = 0.1  # 初始夹爪宽度
        self.max_gripper_width = 0.1
        self.min_gripper_width = 0.0

    def publish_action(self, action):
        """
        模拟发布动作到机器人。
        在真实场景中，这里会调用机器人的API。
        """
        gripper_signal = action[6]
        gripper_speed = 0.01  # 模拟夹爪开合速度
        self.gripper_width += gripper_signal * gripper_speed * 0.1  # 乘以一个小的 dt
        self.gripper_width = np.clip(self.gripper_width, self.min_gripper_width, self.max_gripper_width)

        trans_part = f"Pos: [{action[0]: .4f}, {action[1]: .4f}, {action[2]: .4f}]"
        rot_part = f"Rot: [{action[3]: .4f}, {action[4]: .4f}, {action[5]: .4f}]"
        gripper_part = f"Gripper: {self.gripper_width:.3f}"

        print(f"\r发布动作 -> {trans_part} | {rot_part} | {gripper_part}", end="")


class RobotController:
    """
    从共享内存读取动作并发送给机器人
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=200):
        self.robot = MockRobot()
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.frequency = frequency
        self.shm = None
        self.action_buffer = None
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
        rate = 1.0 / self.frequency
        try:
            while not self.stop_event.is_set():
                start_time = time.time()

                action = self.action_buffer.copy()
                self.robot.publish_action(action)

                elapsed_time = time.time() - start_time
                sleep_time = rate - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("\n机器人控制器：正在关闭...")
            if self.shm:
                self.shm.close()

    def stop(self):
        self.stop_event.set()


# --- 4. 主程序入口 ---

def main():
    # --- 实时配置参数 ---
    ACTION_SHM_NAME = 'teleop_action_buffer'
    ACTION_DIM = 7
    ACTION_DTYPE = np.float64
    TELEOP_FREQUENCY = 100  # Hz
    ROBOT_FREQUENCY = 200  # Hz

    # 定义共享内存的规格
    shm_shape = (ACTION_DIM,)
    shm_size = np.prod(shm_shape) * np.dtype(ACTION_DTYPE).itemsize

    # 创建共享内存块
    try:
        shm = shared_memory.SharedMemory(name=ACTION_SHM_NAME, create=True, size=shm_size)
        print(f"主程序：成功创建共享内存 '{ACTION_SHM_NAME}'，大小 {shm_size} 字节。")
    except FileExistsError:
        print(f"主程序：共享内存 '{ACTION_SHM_NAME}' 已存在，将连接到它。")
        shm = shared_memory.SharedMemory(name=ACTION_SHM_NAME)

    # 初始化遥操作设备和机器人控制器，并传入配置
    teleop_device = KeyboardTeleop(
        shm_name=ACTION_SHM_NAME,
        shm_shape=shm_shape,
        shm_dtype=ACTION_DTYPE,
        frequency=TELEOP_FREQUENCY
    )
    robot_controller = RobotController(
        shm_name=ACTION_SHM_NAME,
        shm_shape=shm_shape,
        shm_dtype=ACTION_DTYPE,
        frequency=ROBOT_FREQUENCY
    )

    # 为每个组件创建一个独立的进程
    teleop_process = mp.Process(target=teleop_device.run)
    robot_process = mp.Process(target=robot_controller.run)

    try:
        print("\n--- 控制说明 ---")
        print("平移: W/S (前/后), A/D (左/右), Q/E (上/下)")
        print("旋转: U/O (绕X轴), I/K (绕Y轴), J/L (绕Z轴)")
        print("夹爪: 按住 空格键 闭合, 松开 张开")
        print("按 Ctrl+C 退出程序。")
        print("-----------------\n")

        teleop_process.start()
        robot_process.start()

        teleop_process.join()
        robot_process.join()

    except KeyboardInterrupt:
        print("\n主程序：检测到 Ctrl+C，正在请求所有进程停止...")
    finally:
        teleop_device.stop()
        robot_controller.stop()

        if teleop_process.is_alive():
            teleop_process.join()
        if robot_process.is_alive():
            robot_process.join()

        print("主程序：所有进程已停止。")

        shm.close()
        shm.unlink()
        print(f"主程序：共享内存 '{ACTION_SHM_NAME}' 已被清理。")


if __name__ == '__main__':
    main()
