import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Dict, Any, List

# 假设 base_robot.py 在同一个目录下
from deploy.robot.base import BaseRobot, RateLimiter


class DeltaEERobot(BaseRobot):
    """
    一个基于PyBullet的机器人实现，通过末端执行器（EE）的增量进行控制。

    这个类封装了PyBullet的交互逻辑，实现了与仿真机器人的连接、
    观测获取、动作发布和关闭等功能。
    """

    def __init__(self,
                 robot_urdf_path: str="franka_panda/panda.urdf",
                 ee_link_index: int=8,
                 initial_joint_positions: tuple=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], gripper_joint_indices: List[int]=[9, 10], gripper_width: float=0.04,
                 use_gui: bool = True, extra_arg=None, *args, **kwargs):
        """
        初始化PyBullet机器人环境。

        Args:
            robot_urdf_path (str): 机器人URDF文件的路径。
            ee_link_index (int): 末端执行器（EE）的link索引。
            initial_joint_positions (tuple): 机器人关节的初始位置。
            use_gui (bool, optional): 是否启动PyBullet的图形界面。默认为 True。
        """
        self.use_gui = use_gui
        self.physics_client = None
        self.robot_id = None

        # 机器人相关参数
        self._robot_urdf_path = robot_urdf_path
        self._ee_link_index = ee_link_index
        self._initial_joint_positions = initial_joint_positions
        self.gripper_joint_indices = gripper_joint_indices
        self.gripper_width = gripper_width
        # 获取可动的关节索引
        _temp_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        _temp_robot_id = p.loadURDF(self._robot_urdf_path, useFixedBase=True, physicsClientId=_temp_client)
        self._num_joints = p.getNumJoints(_temp_robot_id, physicsClientId=_temp_client)
        self._controllable_joints = [i for i in range(self._num_joints) if p.getJointInfo(
            _temp_robot_id, i, physicsClientId=_temp_client)[2] != p.JOINT_FIXED]
        p.disconnect(physicsClientId=_temp_client)

        # 视觉相关参数
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.5],
            distance=1.5,
            yaw=90,
            pitch=-20,
            roll=0,
            upAxisIndex=2
        )
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )

        # 速率控制器
        self._rate_limiter = RateLimiter()

    def connect(self):
        """连接到PyBullet物理服务器并加载机器人。"""
        if self.physics_client is not None:
            print("Already connected to PyBullet.")
            return True
        try:
            print("Connecting to PyBullet...")
            client_mode = p.GUI if self.use_gui else p.DIRECT
            self.physics_client = p.connect(client_mode)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # 加载地面和机器人
            p.loadURDF("plane.urdf")
            self.robot_id = p.loadURDF(self._robot_urdf_path, [0, 0, 0], useFixedBase=True)

            # 重置机器人到初始姿态
            for i, pos in zip(self._controllable_joints, self._initial_joint_positions):
                p.resetJointState(self.robot_id, i, pos)

            print(f"Robot '{self._robot_urdf_path}' loaded with ID: {self.robot_id}")
        except Exception as e:
            print(f"Failed to connect to PyBullet: {e}")
            return False


    def get_observation(self) -> Dict[str, Any]:
        """
        从PyBullet获取多模态观测数据。

        Returns:
            Dict[str, Any]: 包含关节位置和图像的观测字典。
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running. Call connect() first.")

        # 1. 获取关节位置 (qpos)
        joint_states = p.getJointStates(self.robot_id, self._controllable_joints)
        qpos = np.array([state[0] for state in joint_states])

        # 2. 获取图像
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        # PyBullet返回RGBA，我们需要转换为RGB并重塑
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # 丢弃 alpha 通道

        # 组装观测字典，以匹配基类中 obs2meta 的期望格式
        obs = {
            'qpos': qpos,
            'image': {
                'front_camera': rgb_array
            }
        }
        return obs

    def publish_action(self, action: np.ndarray):
        """
        发布一个新的动作来控制机器人。

        Args:
            action (np.ndarray): 一个3D或6D的向量。
                                 - action[:3]: 期望的EE位置变化 (dx, dy, dz)。
                                 - action[3:6]: (可选) 期望的EE姿态变化 (d_roll, d_pitch, d_yaw)。
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running.")

        # --- 位置控制 (与之前相同) ---
        delta_pos = action[:3]
        ee_state = p.getLinkState(self.robot_id, self._ee_link_index, computeForwardKinematics=True)
        current_pos = np.array(ee_state[4])
        current_orn_quat = np.array(ee_state[5])  # 当前姿态是一个四元数 [x, y, z, w]
        target_pos = current_pos + delta_pos

        # --- 姿态控制 (新增部分) ---
        # 默认目标姿态为当前姿态
        target_orn_quat = current_orn_quat

        # 如果action包含姿态信息
        if len(action) >= 6:
            delta_orn_euler = action[3:6]

            # 1. 将欧拉角增量转换为四元数增量
            delta_orn_quat = p.getQuaternionFromEuler(delta_orn_euler)

            # 2. 将当前姿态四元数与增量四元数相乘，得到目标姿态
            # p.multiplyTransforms 返回组合后的 (位置, 姿态)
            # 我们只需要姿态部分，即索引为1的结果
            _, target_orn_quat = p.multiplyTransforms(
                positionA=[0, 0, 0],  # 位置不重要
                orientationA=current_orn_quat,  # 基础姿态
                positionB=[0, 0, 0],  # 位置不重要
                orientationB=delta_orn_quat  # 要应用的旋转
            )
            target_gripper_pos = self.gripper_width * action[-1]



        # --- 逆运动学 (IK) ---
        # 使用新的目标位置和目标姿态
        target_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self._ee_link_index,
            targetPosition=target_pos,
            targetOrientation=target_orn_quat,  # <--- 使用计算出的目标姿态
            # 添加一些求解器参数可以提高稳定性和成功率
            solver=0,
            maxNumIterations=100,
            residualThreshold=.01
        )

        # --- 电机控制 (与之前相同) ---
        MAX_FORCE = 100
        for i, joint_id in enumerate(self._controllable_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_positions[i],
                force=MAX_FORCE,
            )
        # 为每个抓夹关节设置目标位置
        for joint_id in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_gripper_pos,
                force=20  # 抓夹的力量可以小一些
            )
        p.stepSimulation()

    def is_running(self) -> bool:
        """检查PyBullet仿真是否仍在运行。"""
        return self.physics_client is not None and p.isConnected(self.physics_client)

    def shutdown(self):
        """断开与PyBullet服务器的连接。"""
        if self.is_running():
            print("Disconnecting from PyBullet...")
            p.disconnect(self.physics_client)
            self.physics_client = None