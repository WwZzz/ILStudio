"""
SO101++ (so101_pp) Follower Robot — parallel gripper variant of so101_plus.
"""

import numpy as np
import traceback
import time
from typing import Optional, List
from pathlib import Path

try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError

from deploy.robot.base import BaseRobot
from .config import SO101PPConfig
from .so101_pp import SO101PP, JOINT_NAMES

DEFAULT_QLIMIT_MIN = [-2.1, -3.1, -0.0, -1.375, -1.57, -1.57, -0.15]
DEFAULT_QLIMIT_MAX = [2.1, 0.0, 3.1, 1.475, 3.1, 1.57, 1.5]
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1, 1]

N_BODY_JOINTS = 6
N_JOINTS = 7


class So101PP(BaseRobot):
    """
    ILStudio wrapper around SO101PP.

    Observation / action qpos order (7D):
      [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, wrist_yaw, gripper]
    """

    CONTROL_MODES = ("qpos",)

    def __init__(
        self,
        name: str = "so101_pp",
        max_size_mb: int = 64,
        fps: float = 100.0,
        control_shm_name: Optional[str] = None,
        com: str = "/dev/ttyACM0",
        robot_id: str = "so101_pp_arm",
        camera_configs: dict = {},
        calibration_dir: Optional[str] = None,
        control_mode: str = "qpos",
        qlimit_min: Optional[List[float]] = None,
        qlimit_max: Optional[List[float]] = None,
        joint_signs: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)

        self.com = com
        self.robot_id = robot_id

        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode

        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS)

        if len(self.qlimit_min) != N_JOINTS or len(self.qlimit_max) != N_JOINTS:
            raise ValueError(f"qlimit_min/max must be length {N_JOINTS}")
        if len(self.joint_signs) != N_JOINTS:
            raise ValueError(f"joint_signs must be length {N_JOINTS}")

        print(f"[So101PP] Joints: {list(JOINT_NAMES)}")
        print(f"[So101PP] Control mode: {control_mode}")

        robot_config = SO101PPConfig(port=com, id=robot_id, cameras={})
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir).expanduser()
        self._robot = SO101PP(robot_config)
        self._motors = list(self._robot.bus.motors)

        if len(self._motors) != N_JOINTS:
            raise RuntimeError(f"Expected {N_JOINTS} motors, got {len(self._motors)}: {self._motors}")

        retry_counts = 0
        max_connect_retry = 10
        while not self.connect():
            print(f"Retrying for {retry_counts} time...")
            retry_counts += 1
            if retry_counts > max_connect_retry:
                raise RuntimeError("Failed to connect to robot after max retries")
            time.sleep(1)

    def connect(self):
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Robot already connected: {e}")
        except Exception as e:
            print(f"Failed to connect to robot due to {e}")
            traceback.print_exc()
            return False
        print("Robot connected")
        print(f"  motors: {[(m, self._robot.bus.motors[m].id) for m in self._motors]}")
        print(f"  calibration file: {self._robot.calibration_fpath}")
        return True

    def get_action_dim(self):
        return len(self._motors)

    def get_observation(self):
        try:
            obs = self._robot.get_observation()
            qpos = np.array([obs[mname + ".pos"] for mname in self._motors], dtype=np.float64)
            return {"qpos": qpos, **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None

    def process_action(self, action_dict: dict) -> dict:
        return action_dict

    def obs2meta(self, device_data):
        if device_data is None:
            return {}
        qpos = device_data.get("qpos")
        if qpos is None:
            qpos = np.array([device_data[m + ".pos"] for m in self._motors], dtype=np.float32)
        return {"state": np.asarray(qpos, dtype=np.float32)}

    def shutdown(self):
        if self._robot.is_connected:
            self._robot.disconnect()

    def close(self):
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()

    def publish_action(self, action: np.ndarray):
        try:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if len(action) != N_JOINTS:
                print(f"[So101PP] Warning: expected {N_JOINTS}D action, got {len(action)}D")
                return
            action_dict = {mname + ".pos": action[i] for i, mname in enumerate(self._motors)}
            self._robot.send_action(action_dict)
        except Exception:
            pass

    def is_running(self):
        return self._robot.is_connected
