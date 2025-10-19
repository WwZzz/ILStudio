"""
So101 Leader 遥操作实现
参照 Koch Leader 的实现方式
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from .bi_so101_leader import BiSO101LeaderConfig as RobotBiSO101LeaderConfig, BiSO101Leader as RobotBiSO101Leader
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
from typing import Optional
from pathlib import Path
    

class BiSO101Leader(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using So101 Leader
    参照 Koch Leader 的实现方式
    """
    def __init__(self, 
                shm_name: str, 
                shm_shape: tuple, 
                shm_dtype: type, 
                action_dim: int = 12,
                action_dtype = np.float32,
                frequency: int = 100, 
                left_arm_port: str="/dev/ttyACM1",    
                right_arm_port: str="/dev/ttyACM2",    
                robot_id: str="bi_so101_leader_arm",
                calibration_dir: Optional[str]=None,
                **kwargs):
        """
        Initialize the BiSO101 Leader teleoperation device
        
        Args:
            shm_name: Name of the shared memory segment
            shm_shape: Shape of the shared memory array
            shm_dtype: Data type of the shared memory array
            action_dim: The dim of the flattened action
            frequency: Control frequency in Hz
            left_arm_port: Communication port for the left arm
            right_arm_port: Communication port for the right arm
            robot_id: Identifier for the robot
        """
        super().__init__(shm_name, shm_shape, shm_dtype, action_dim, action_dtype, frequency)
        
        self.left_arm_port = left_arm_port
        self.right_arm_port = right_arm_port
        self.robot_id = robot_id
        
        # Use the official lerobot support:
        robot_config = RobotBiSO101LeaderConfig(left_arm_port=left_arm_port, right_arm_port=right_arm_port, id=robot_id)
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._teleop_device = RobotBiSO101Leader(robot_config)
        self._teleop_device.connect()
        self._left_motors = list(self._teleop_device.left_arm.bus.motors)
        self._right_motors = list(self._teleop_device.right_arm.bus.motors)
        
    def get_observation(self):
        """Get the observation data for the Leader device"""
        return self._teleop_device.get_action()
    
    def observation_to_action(self, observation):
        """Convert the observation data to the standardized robot action"""
        left_qpos = np.array([observation['left_'+mname+'.pos'] for mname in self._left_motors], dtype=self.action_dtype)
        right_qpos = np.array([observation['right_'+mname+'.pos'] for mname in self._right_motors], dtype=self.action_dtype)
        return np.concatenate([left_qpos, right_qpos])
    
    def stop(self):
        """Stop the teleoperation device"""
        if self._teleop_device.is_connected:
            self._teleop_device.disconnect()
    
    def get_doc(self) -> str:
        """Get the documentation for the device"""
        return """
        BiSO101 Leader Teleoperation Device
        
        This device allows you to control a BiSO101 follower robot by manipulating
        a BiSO101 leader device. The leader device captures your hand movements
        and translates them into commands for the follower robot.
        
        Controls:
        - Move the leader robot joints to control the follower
        - The system captures joint positions and sends them to the follower
        
        Configuration:
        - left_arm_port: Communication port for the left arm (default: /dev/ttyACM1)
        - right_arm_port: Communication port for the right arm (default: /dev/ttyACM2)
        - robot_id: Identifier for the robot (default: bi_so101_leader_arm)
        - action_dim: Number of controllable joints (default: 12)
        - frequency: Control frequency in Hz (default: 100)
        """
