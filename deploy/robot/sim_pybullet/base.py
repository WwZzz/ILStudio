import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import Dict, Any

# Assume base_robot.py is in the same directory
from deploy.robot.base import BaseRobot, RateLimiter


class PyBulletDeltaEERobot(BaseRobot):
    """
    A PyBullet-based robot implementation controlled via incremental
    end-effector (EE) commands.

    This class encapsulates PyBullet interaction logic and provides
    functionality for connecting to the simulated robot, retrieving
    observations, publishing actions, and shutting down.
    """

    def __init__(self,
                 robot_urdf_path: str,
                 ee_link_index: int,
                 initial_joint_positions: tuple,
                 use_gui: bool = True, extra_arg=None, *args, **kwargs):
        """
        Initialize the PyBullet robot environment.

        Args:
            robot_urdf_path (str): Path to the robot's URDF file.
            ee_link_index (int): Link index of the end-effector (EE).
            initial_joint_positions (tuple): Initial positions for the robot joints.
            use_gui (bool, optional): Whether to launch PyBullet's GUI. Defaults to True.
        """
        self.use_gui = use_gui
        self.physics_client = None
        self.robot_id = None

        # Robot-related parameters
        self._robot_urdf_path = robot_urdf_path
        self._ee_link_index = ee_link_index
        self._initial_joint_positions = initial_joint_positions

        # Obtain indices of actuated joints
        # Temporarily load the robot to determine the number of joints and their indices
        _temp_client = p.connect(p.DIRECT)
        _temp_robot_id = p.loadURDF(self._robot_urdf_path, useFixedBase=True, physicsClientId=_temp_client)
        self._num_joints = p.getNumJoints(_temp_robot_id, physicsClientId=_temp_client)
        self._controllable_joints = [i for i in range(self._num_joints) if p.getJointInfo(
            _temp_robot_id, i, physicsClientId=_temp_client)[2] != p.JOINT_FIXED]
        p.disconnect(physicsClientId=_temp_client)

        # Vision-related parameters
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

        # Rate limiter
        self._rate_limiter = RateLimiter()

    def connect(self):
        """Connect to the PyBullet physics server and load the robot."""
        if self.physics_client is not None:
            print("Already connected to PyBullet.")
            return

        print("Connecting to PyBullet...")
        client_mode = p.GUI if self.use_gui else p.DIRECT
        self.physics_client = p.connect(client_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground plane and robot
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self._robot_urdf_path, [0, 0, 0], useFixedBase=True)

        # Reset the robot to its initial pose
        for i, pos in zip(self._controllable_joints, self._initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)

        print(f"Robot '{self._robot_urdf_path}' loaded with ID: {self.robot_id}")

    def get_observation(self) -> Dict[str, Any]:
        """
        Retrieve multimodal observation data from PyBullet.

        Returns:
            Dict[str, Any]: Observation dictionary containing joint positions and images.
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running. Call connect() first.")

        # 1. Get joint positions (qpos)
        joint_states = p.getJointStates(self.robot_id, self._controllable_joints)
        qpos = np.array([state[0] for state in joint_states])

        # 2. Retrieve image
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        # PyBullet returns RGBA, convert to RGB and reshape
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array.reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]  # Drop alpha channel

        # Assemble observation dictionary to match expected format in base class obs2meta
        obs = {
            'qpos': qpos,
            'image': {
                'front_camera': rgb_array
            }
        }
        return obs

    def publish_action(self, action: np.ndarray):
        """
        Publish an action command to control the robot.

        Args:
            action (np.ndarray): A 3D or 6D vector representing EE position change (dx, dy, dz, ...).
                                 Here we only use the first three positional components.
        """
        if not self.is_running():
            raise ConnectionError("PyBullet is not running.")

        delta_pos = action[:3]  # Only take positional delta

        # Get current EE state
        ee_state = p.getLinkState(self.robot_id, self._ee_link_index, computeForwardKinematics=True)
        current_pos = np.array(ee_state[4])
        current_orn = np.array(ee_state[5])  # Keep current orientation unchanged

        # Compute target position
        target_pos = current_pos + delta_pos

        # Use inverse kinematics (IK) to compute target joint angles
        target_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self._ee_link_index,
            targetPosition=target_pos,
            targetOrientation=current_orn,
            # solver=p.IK_SDLS,  # Optional IK solver
            # restPoses=list(self._initial_joint_positions),  # Helps resolve multiple solutions
        )

        # Send joint position control commands
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndices=self._controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            # maxVelocity=...  # Optionally limit max velocity
            # force=...        # Optionally limit max force/torque
        )

        # In PyBullet, after sending commands, step the simulation to make them effective
        p.stepSimulation()

    def is_running(self) -> bool:
        """Check whether the PyBullet simulation is still running."""
        return self.physics_client is not None and p.isConnected(self.physics_client)

    def rate_sleep(self, hz: int):
        """Sleep at the specified frequency."""
        self._rate_limiter.sleep(hz)

    def shutdown(self):
        """Disconnect from the PyBullet server."""
        if self.is_running():
            print("Disconnecting from PyBullet...")
            p.disconnect(self.physics_client)
            self.physics_client = None