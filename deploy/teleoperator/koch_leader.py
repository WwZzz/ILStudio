from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig as RobotKochLeaderConfig, KochLeader as RobotKochLeader
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

class KochLeader(BaseTeleopDevice):
    """
    Concrete implementation of teleoperation using lerobot's KochLeader
    """
    def __init__(self, 
                shm_name: str, 
                shm_shape: tuple, 
                shm_dtype: type, 
                action_dim: int = 6,
                action_dtype = np.float32,
                frequency: int = 100, 
                com: str="COM7",
                robot_id: str="koch_leader_arm",
                elbow_drive_mode: int=1,
            ):
        """
        Initialize the keyboard teleoperation device
        
        Args:
            shm_name: Name of the shared memory segment
            shm_shape: Shape of the shared memory array
            shm_dtype: Data type of the shared memory array
            action_dim: The dim of the flattened action
            frequency: Control frequency in Hz
            gripper_index: Index of the gripper control in the action array
            gripper_width: Maximum width of the gripper in meters
        """
        super().__init__(shm_name, shm_shape, shm_dtype, action_dim, action_dtype, frequency)
        self._teleop_device = RobotKochLeader(RobotKochLeaderConfig(port=com, id=robot_id))
        self.elbow_drive_mode = elbow_drive_mode
        self._teleop_device.connect()
        self._teleop_device.bus.write("Drive_Mode", "elbow_flex", self.elbow_drive_mode)
        self._motors = list(self._teleop_device.bus.motors)
        
    def get_observation(self):
        return self._teleop_device.get_action()
    
    def observation_to_action(self, observation):
        return np.array([observation[mname+'.pos'] for mname in self._motors], dtype=self.action_dtype)
    
    def stop(self):
        if self._teleop_device.is_connected:
            self._teleop_device.disconnect()
