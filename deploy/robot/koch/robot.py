"""
集成相机的 KochFollower 机器人实现
将相机直接集成到 lerobot 的默认 KochFollower 中
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower
from deploy.robot.base import BaseRobot
import numpy as np
import traceback
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from benchmark.base import MetaAction, MetaObs

class KochFollowerWithCamera(BaseRobot):
    """
    将相机直接集成到 lerobot 的默认 KochFollower 中
    这样相机就成为机器人的一部分，而不是外部组件
    """
    def __init__(self, com: str="COM8", robot_id: str="koch_follower_arm", camera_configs: dict={}, **kwargs):
        import threading, queue
        super().__init__()
        
        # 创建相机配置
        camera_configs_dict = {}
        for cam_name, cam_config in camera_configs.items():
            camera_configs_dict[cam_name] = OpenCVCameraConfig(**cam_config)
        
        # 机械臂部分 - 直接传递相机配置给 KochFollower
        self._robot = KochFollower(KochFollowerConfig(
            port=com, 
            id=robot_id, 
            cameras=camera_configs_dict  # 将相机配置传递给 lerobot 的默认实现
        ))
        
        self._motors = list(self._robot.bus.motors)
    
    def connect(self):
        """连接机器人和相机"""
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
        """获取动作维度"""
        return len(self._motors)

    def get_observation(self):
        """获取完整观测数据（包括相机图像）"""
        # 直接使用 KochFollower 的 get_observation 方法
        # 它已经包含了相机图像数据
        try:
            obs = self._robot.get_observation()
            return obs
        except Exception as e:
            # 返回默认观测数据
            return None

    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        obs['qpos'] = np.array([obs[mname+'.pos'] for mname in self._motors], dtype=np.float32)
        return MetaObs(state=obs['qpos'], state_joint=obs['qpos'], image=obs['front_camera'][np.newaxis,:].transpose(0, 3, 1, 2))
    
    def shutdown(self):
        """关闭机器人和相机"""
        if self._robot.is_connected:
            self._robot.disconnect()
        # 相机通过机器人的 disconnect() 方法自动关闭
        
    def publish_action(self, action: np.ndarray):
        """发布动作到机器人"""
        try:
            action_dict = {mname+'.pos': action[i] for i, mname in enumerate(self._motors)}
            self._robot.send_action(action_dict)
        except Exception as e:
            pass
            # print(f"Warning: Failed to publish action: {e}")
    
    def is_running(self):
        """检查机器人是否正在运行"""
        return self._robot.is_connected

    def save_episode(self, file_path: str, observations: list, actions: list):
        """保存episode数据到HDF5文件"""
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
