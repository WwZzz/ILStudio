"""
So101 Leader 遥操作实现
参照 Koch Leader 的实现方式
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig as RobotSO101LeaderConfig, SO101Leader as RobotSO101Leader
from deploy.robot.base import BaseRobot
import numpy as np
import traceback
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard
from deploy.teleoperator.base import BaseTeleopDevice


class So101Leader(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using So101 Leader
    参照 Koch Leader 的实现方式
    """
    def __init__(self, 
                shm_name: str, 
                shm_shape: tuple, 
                shm_dtype: type, 
                action_dim: int = 6,
                action_dtype = np.float32,
                frequency: int = 100, 
                com: str="COM7",    
                robot_id: str="so101_leader_arm",
                elbow_drive_mode: int=1,
            ):
        """
        Initialize the So101 Leader teleoperation device
        
        Args:
            shm_name: Name of the shared memory segment
            shm_shape: Shape of the shared memory array
            shm_dtype: Data type of the shared memory array
            action_dim: The dim of the flattened action
            frequency: Control frequency in Hz
            com: Communication port for the leader device
            robot_id: Identifier for the robot
            elbow_drive_mode: Drive mode for elbow joint
        """
        super().__init__(shm_name, shm_shape, shm_dtype, action_dim, action_dtype, frequency)
        
        self.com = com
        self.robot_id = robot_id
        self.elbow_drive_mode = elbow_drive_mode
        
        # 使用官方的 lerobot 支持：
        self._teleop_device = RobotSO101Leader(RobotSO101LeaderConfig(port=com, id=robot_id))
        self._teleop_device.connect()
        # self._teleop_device.bus.write("Drive_Mode", "elbow_flex", self.elbow_drive_mode)
        self._motors = list(self._teleop_device.bus.motors)
        
        
        # 备用实现（如果官方支持有问题）：
        # self._teleop_device = None  # 占位符
        # self._motors = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        # self._is_connected = False
        # self._current_observation = np.zeros(action_dim, dtype=action_dtype)
        # self._connect_device()
        
        
    def get_observation(self):
        """获取 Leader 设备的观测数据"""
        return self._teleop_device.get_action()
    
    def observation_to_action(self, observation):
        """将观测数据转换为标准化的机器人动作"""
        return np.array([observation[mname+'.pos'] for mname in self._motors], dtype=self.action_dtype)
    
    def stop(self):
        """停止遥操作设备"""
        if self._teleop_device.is_connected:
            self._teleop_device.disconnect()
    
    def get_doc(self) -> str:
        """获取设备文档"""
        return """
        So101 Leader Teleoperation Device
        
        This device allows you to control a So101 follower robot by manipulating
        a So101 leader device. The leader device captures your hand movements
        and translates them into commands for the follower robot.
        
        Controls:
        - Move the leader robot joints to control the follower
        - The system captures joint positions and sends them to the follower
        
        Configuration:
        - com: Communication port (default: COM7 on Windows, /dev/ttyACM0 on Linux)
        - action_dim: Number of controllable joints (default: 6)
        - frequency: Control frequency in Hz (default: 100)
        """
