"""
Implement So101Follower robot with camera integration
Follow the implementation of Koch robot, provide a unified interface for So101 robot
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from .bi_so101_follower import BiSO101FollowerConfig, BiSO101Follower
from deploy.robot.base import BaseRobot
from lerobot.robots.robot import Robot
import numpy as np
import traceback
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from benchmark.base import MetaAction, MetaObs
from pathlib import Path
from typing import Optional

class BiSo101FollowerWithCamera(BaseRobot):
    """
    Integrate camera directly into BiSO101Follower
    So that the camera becomes part of the robot, rather than an external component
    
    Note: Since lerobot may not directly support BiSO101, this provides a basic framework
    Need to adjust according to the actual BiSO101 SDK
    """
    def __init__(self, left_arm_port: str="/dev/ttyACM0", right_arm_port: str="/dev/ttyACM3", robot_id: str="bi_so101_follower_arm", camera_configs: dict={}, calibration_dir: Optional[str]=None, **kwargs):
        super().__init__()
        
        self.left_arm_port = left_arm_port
        self.right_arm_port = right_arm_port
        self.robot_id = robot_id
        
        # Create camera configurations
        self.camera_configs_dict = {}
        for cam_name, cam_config in camera_configs.items():
            self.camera_configs_dict[cam_name] = OpenCVCameraConfig(**cam_config)
        
        # Initialize cameras
        self.cameras = {}
        for cam_name, cam_config in self.camera_configs_dict.items():
            self.cameras[cam_name] = OpenCVCamera(cam_config)
        
        # So101 arm part - using official lerobot support
        robot_config = BiSO101FollowerConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            id=robot_id, 
            cameras=self.camera_configs_dict  # Pass camera configurations to lerobot's default implementation
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._robot = BiSO101Follower(robot_config)
        self._left_motors = list(self._robot.left_arm.bus.motors)
        self._right_motors = list(self._robot.right_arm.bus.motors)
    
    def connect(self):
        """Connect robot and cameras"""
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Robot already connected: {e}")
            pass
        except Exception as e:
            print(f"Failed to connect to robot due to {e}")
            traceback.print_exc()
            return False
        print("Robot connected")
        return True
    
    def get_action_dim(self):
        """Get action dimension"""
        return len(self._left_motors) + len(self._right_motors)

    def get_observation(self):
        """Get complete observation data (including camera images)"""
        # Directly use SO101Follower's get_observation method
        # It already contains camera image data
        try:
            obs = self._robot.get_observation()
            return obs
        except Exception as e:
            # Return default observation data
            return None

    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        if obs is None:
            return None
        left_qpos = np.array([obs['left_'+mname+'.pos'] for mname in self._left_motors], dtype=np.float32)
        right_qpos = np.array([obs['right_'+mname+'.pos'] for mname in self._right_motors], dtype=np.float32)
        obs['qpos'] = np.concatenate([left_qpos, right_qpos])
        
        # Process image data
        if 'front_camera' in obs:
            image = obs['front_camera'][np.newaxis,:].transpose(0, 3, 1, 2)
        else:
            # If no image, create a default empty image
            image = np.zeros((1, 3, 480, 640), dtype=np.uint8)
            
        return MetaObs(state=obs['qpos'], state_joint=obs['qpos'], image=image)
    
    def shutdown(self):
        """Shutdown robot and cameras"""
        if self._robot.is_connected:
            self._robot.disconnect()
        # Cameras are automatically closed by the robot's disconnect() method
        
    def publish_action(self, action: np.ndarray):
        """Publish action to robot"""
        try:
            left_action = {'left_'+mname+'.pos': action[i] for i, mname in enumerate(self._left_motors)}
            right_action = {'right_'+mname+'.pos': action[i+len(self._left_motors)] for i, mname in enumerate(self._right_motors)}
            action_dict = {**left_action, **right_action}
            self._robot.send_action(action_dict)
        except Exception as e:
            pass
            # print(f"Warning: Failed to publish action: {e}")
    
    def is_running(self):
        """Check if robot is running"""
        return self._robot.is_connected

    def save_episode(self, file_path: str, observations: list, actions: list):
        """Save episode data to HDF5 file"""
        import h5py
        import os
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        def write_group(group, data_list, key_prefix=None):
            # data_list: list of dict or value
            if isinstance(data_list[0], dict):
                # For each key, collect list of values and recurse
                for key in data_list[0].keys():
                    sub_list = [obs[key] for obs in data_list]
                    if isinstance(sub_list[0], dict):
                        sub_group = group.create_group(key)
                        write_group(sub_group, sub_list)
                    else:
                        try:
                            group.create_dataset(key, data=np.stack(sub_list))
                        except (TypeError, ValueError) as e:
                            print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
            else:
                # If not dict, just create dataset
                try:
                    if key_prefix is None:
                        group.create_dataset('data', data=np.stack(data_list))
                    else:
                        group.create_dataset(key_prefix, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            obs_group = f.create_group('observations')
            if observations:
                write_group(obs_group, observations)
