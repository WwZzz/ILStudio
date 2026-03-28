"""
Bessica-D dual-arm simulation robot (MuJoCo + Pinocchio IK).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np

try:
    from .kinematics import BessicaBimanualKinematics, T_to_pose_xyz_rpy
except ImportError:
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from deploy.robot.bessica_sim.kinematics import BessicaBimanualKinematics, T_to_pose_xyz_rpy

from deploy.simulation.mujoco.base import MujocoDeviceBase


ARM_JOINT_NAMES: List[str] = [f"right_arm_joint{i}" for i in range(1, 8)] + [
    f"left_arm_joint{i}" for i in range(1, 8)
]
GRIPPER_JOINT_NAMES: List[str] = [
    "right_arm_gripper_joint",
    "left_arm_gripper_joint",
]
GRIPPER_MIRROR_JOINT_NAMES: List[str] = [
    "right_arm_gripper_mirror_joint",
    "left_arm_gripper_mirror_joint",
]
JOINT_NAMES: List[str] = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

ARM_QLIMIT_MIN = np.full(14, -2.14)
ARM_QLIMIT_MAX = np.full(14, 2.14)

GRIPPER_MAX_WIDTH = 0.101
GRIPPER_MIN_WIDTH = 0.0
# Default onboard cameras defined in mujoco_model/bessica_d.xml (not scene XML).
DEFAULT_BESSICA_CAMERA_NAMES: List[str] = ["leftview", "rightview", "headview"]
DEFAULT_INIT_ARM_QPOS = np.array(
    [
        0.0,
        0.0,
        -1.4,
        1.55,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.4,
        1.55,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def _draw_viewer_frame(viewer, pos: np.ndarray, axis_length: float = 0.05, axis_radius: float = 0.002):
    axes = [
        (np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0, 0.6)),
        (np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0, 0.6)),
        (np.array([0.0, 0.0, 1.0]), (0.0, 0.0, 1.0, 0.6)),
    ]
    for axis, color in axes:
        start = pos
        end = pos + axis * axis_length
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(color, dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, axis_radius, start, end)
        viewer.user_scn.ngeom += 1


class BessicaSimRobot(MujocoDeviceBase):
    CONTROL_MODES = ("delta_ee", "qpos")
    QPOS_INPUT_UNITS = ("normalized", "radians")

    def __init__(
        self,
        name: str = "bessica_sim",
        max_size_mb: int = 64,
        fps: float = 50.0,
        control_shm_name: Optional[str] = None,
        xml_path: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_xml_path: Optional[str] = None,
        camera_names: Optional[List[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        urdf_path: Optional[str] = None,
        control_mode: str = "delta_ee",
        qpos_input_unit: str = "radians",
        position_scale: float = 0.002,
        rotation_scale: float = 0.015,
        gripper_scale: float = 0.0,
        init_qpos: Optional[np.ndarray] = None,
        ik_fast: bool = True,
        simulation_fps: float = 240.0,
        **kwargs,
    ):
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        if qpos_input_unit not in self.QPOS_INPUT_UNITS:
            raise ValueError(f"Invalid qpos_input_unit '{qpos_input_unit}'. Must be one of {self.QPOS_INPUT_UNITS}")

        self.control_mode = control_mode
        self.qpos_input_unit = qpos_input_unit
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        self.ik_fast = ik_fast
        self.simulation_fps = float(simulation_fps)
        self.kin = BessicaBimanualKinematics(urdf_path=urdf_path)
        self._physics_thread: Optional[threading.Thread] = None
        self._physics_stop = threading.Event()

        init_qpos_arr = None if init_qpos is None else np.array(init_qpos, dtype=np.float64).ravel()
        if init_qpos_arr is not None and len(init_qpos_arr) == 16:
            self.current_arm_qpos = np.concatenate([init_qpos_arr[:7], init_qpos_arr[8:15]])
            self.current_gripper_width = np.array([init_qpos_arr[7], init_qpos_arr[15]], dtype=np.float64)
        elif init_qpos_arr is not None:
            self.current_arm_qpos = init_qpos_arr[:14].copy()
            self.current_gripper_width = np.array([GRIPPER_MAX_WIDTH, GRIPPER_MAX_WIDTH], dtype=np.float64)
        else:
            self.current_arm_qpos = DEFAULT_INIT_ARM_QPOS.copy()
            self.current_gripper_width = np.array([GRIPPER_MAX_WIDTH, GRIPPER_MAX_WIDTH], dtype=np.float64)

        self.current_gpos = np.zeros(12, dtype=np.float64)
        self.target_gpos = np.zeros(12, dtype=np.float64)

        print(
            f"[BessicaSimRobot] control_mode={control_mode}"
            + (f", qpos_input_unit={qpos_input_unit}" if control_mode == "qpos" else "")
        )

        if camera_names is None:
            camera_names = list(DEFAULT_BESSICA_CAMERA_NAMES)

        super().__init__(
            name=name,
            max_size_mb=max_size_mb,
            fps=fps,
            control_shm_name=control_shm_name,
            xml_path=xml_path,
            scene_name=scene_name,
            scene_xml_path=scene_xml_path,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            **kwargs,
        )
        print("[BessicaSimRobot] Initialized")

    def _default_robot_xml_path(self) -> str:
        return str(Path(__file__).parent / "mujoco_model" / "bessica_d.xml")

    def _build_joint_indices(self) -> None:
        self.arm_qpos_indices = np.array(
            [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self.gripper_qpos_indices = np.array(
            [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in GRIPPER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.arm_qvel_indices = np.array(
            [self.mjmodel.jnt_dofadr[self.mjmodel.joint(n).id] for n in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self.gripper_qvel_indices = np.array(
            [self.mjmodel.jnt_dofadr[self.mjmodel.joint(n).id] for n in GRIPPER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.gripper_mirror_qpos_indices = np.array(
            [self.mjmodel.jnt_qposadr[self.mjmodel.joint(n).id] for n in GRIPPER_MIRROR_JOINT_NAMES],
            dtype=np.int32,
        )
        self.gripper_mirror_qvel_indices = np.array(
            [self.mjmodel.jnt_dofadr[self.mjmodel.joint(n).id] for n in GRIPPER_MIRROR_JOINT_NAMES],
            dtype=np.int32,
        )
        self.gripper_ctrl_indices = np.array(
            [
                self.mjmodel.actuator("right_arm_gripper_joint").id,
                self.mjmodel.actuator("left_arm_gripper_joint").id,
            ],
            dtype=np.int32,
        )

    def _apply_gripper_to_mujoco(self, width: np.ndarray):
        q = width / 2.0
        self.mjdata.qpos[self.gripper_qpos_indices] = q
        self.mjdata.qpos[self.gripper_mirror_qpos_indices] = q
        self.mjdata.ctrl[self.gripper_ctrl_indices] = q

    def _update_gpos_from_q(self, arm_qpos: np.ndarray) -> np.ndarray:
        q_pin = self.kin.q_mujoco_to_pin(arm_qpos)
        gr = T_to_pose_xyz_rpy(self.kin.fk_T(q_pin, "right"))
        gl = T_to_pose_xyz_rpy(self.kin.fk_T(q_pin, "left"))
        return np.concatenate([gr, gl])

    def _apply_initial_state(self) -> None:
        self._apply_gripper_to_mujoco(self.current_gripper_width)
        self.mjdata.qpos[self.arm_qpos_indices] = self.current_arm_qpos
        self.mjdata.qvel[self.arm_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_mirror_qvel_indices] = 0.0
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        self._commanded_arm_qpos = self.current_arm_qpos.copy()
        self._commanded_gripper_width = self.current_gripper_width.copy()
        self._safe_arm_qpos = self.current_arm_qpos.copy()
        self._safe_gripper_width = self.current_gripper_width.copy()
        self.current_gpos = self._update_gpos_from_q(self.current_arm_qpos)
        self.target_gpos = self.current_gpos.copy()

    def _project_to_safe_static_contact_locked(
        self,
        target_arm_q: np.ndarray,
        target_grip_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = np.concatenate([self._safe_arm_qpos, self._safe_gripper_width])
        target = np.concatenate([target_arm_q, target_grip_w])

        def _apply(v: np.ndarray) -> None:
            self.mjdata.qpos[self.arm_qpos_indices] = v[:14]
            self._apply_gripper_to_mujoco(v[14:16])
            self.mjdata.qvel[self.arm_qvel_indices] = 0.0
            self.mjdata.qvel[self.gripper_qvel_indices] = 0.0
            self.mjdata.qvel[self.gripper_mirror_qvel_indices] = 0.0
            mujoco.mj_forward(self.mjmodel, self.mjdata)

        v_out = self._interpolate_pose_avoiding_static_contact_locked(start, target, _apply)
        return v_out[:14].copy(), v_out[14:16].copy()

    def _setup_viewer_overlay_state(self) -> None:
        self._viewer_tcp_site_ids = []
        for site_name in ("tcp_right", "tcp_left"):
            try:
                self._viewer_tcp_site_ids.append(self.mjmodel.site(site_name).id)
            except KeyError:
                continue

    def _draw_viewer_overlay_locked(self, viewer) -> None:
        for site_id in getattr(self, "_viewer_tcp_site_ids", []):
            _draw_viewer_frame(viewer, self.mjdata.site_xpos[site_id].copy(), 0.07, 0.003)

    def get_action_dim(self) -> int:
        return 14 if self.control_mode == "delta_ee" else 16

    def _get_robot_observation_core(self) -> Dict[str, np.ndarray]:
        arm_q = self.current_arm_qpos.copy()
        grip_w = self.current_gripper_width.copy()
        qpos = np.concatenate([arm_q[:7], grip_w[0:1], arm_q[7:14], grip_w[1:2]])
        return {"qpos": qpos, "gpos": self.current_gpos.copy()}

    def _process_robot_action(self, action_dict: dict) -> dict:
        action = action_dict.get("action", None)
        if action is None:
            return action_dict
        action = np.asarray(action, dtype=np.float64)
        if self.control_mode == "qpos":
            return self._process_qpos(action_dict, action)
        return self._process_delta_ee(action_dict, action)

    def _process_qpos(self, action_dict: dict, action: np.ndarray) -> dict:
        if len(action) != 16:
            print(f"[BessicaSimRobot] qpos mode expects 16D action, got {len(action)}D")
            return action_dict

        arm_action = np.concatenate([action[:7], action[8:15]])
        grip_action = np.array([action[7], action[15]], dtype=np.float64)

        if self.qpos_input_unit == "normalized":
            arm_target = (arm_action / 100.0).clip(-1.0, 1.0)
            span = (ARM_QLIMIT_MAX - ARM_QLIMIT_MIN) * 0.5
            mid = (ARM_QLIMIT_MAX + ARM_QLIMIT_MIN) * 0.5
            arm_target = mid + arm_target * span
            grip_target = np.clip(grip_action / 100.0, 0.0, 1.0) * GRIPPER_MAX_WIDTH
        else:
            arm_target = arm_action.copy()
            grip_target = grip_action.copy()

        arm_target = np.clip(arm_target, ARM_QLIMIT_MIN, ARM_QLIMIT_MAX)
        grip_target = np.clip(grip_target, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)
        action_dict["action"] = np.concatenate([arm_target, grip_target])
        return action_dict

    def _process_delta_ee(self, action_dict: dict, action: np.ndarray) -> dict:
        if len(action) != 14:
            print(f"[BessicaSimRobot] delta_ee expects 14D action, got {len(action)}D")
            return action_dict

        d_r = action[0:7].copy()
        d_l = action[7:14].copy()
        grip_r_delta = d_r[6] * self.gripper_scale
        grip_l_delta = d_l[6] * self.gripper_scale

        pos_rot = (
            np.sum(np.abs(d_r[:6]))
            + np.sum(np.abs(d_l[:6]))
            + abs(grip_r_delta)
            + abs(grip_l_delta)
        )
        if pos_rot < 1e-10:
            action_dict["action"] = None
            return action_dict

        with self._lock:
            q_pin = self.kin.q_mujoco_to_pin(self.current_arm_qpos)
            ok, q_new = self.kin.solve_delta_bimanual(
                q_pin,
                d_r[:6],
                d_l[:6],
                self.position_scale,
                self.rotation_scale,
                fast=self.ik_fast,
            )
            q_mj = self.kin.q_pin_to_mujoco(q_new)
            q_mj = np.clip(q_mj, ARM_QLIMIT_MIN, ARM_QLIMIT_MAX)

            new_grip = self.current_gripper_width.copy()
            new_grip[0] = np.clip(new_grip[0] + grip_r_delta, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)
            new_grip[1] = np.clip(new_grip[1] + grip_l_delta, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)

            if ok:
                self.target_gpos = self._update_gpos_from_q(q_mj)
                action_dict["action"] = np.concatenate([q_mj, new_grip])
            else:
                action_dict["action"] = np.concatenate([self.current_arm_qpos.copy(), new_grip])

        return action_dict

    def _publish_robot_action(self, action: np.ndarray) -> None:
        with self._lock:
            arm_q = action[:14]
            grip_w = action[14:16] if len(action) >= 16 else self.current_gripper_width
            arm_q, grip_w = self._project_to_safe_static_contact_locked(arm_q, grip_w)
            self._commanded_arm_qpos = arm_q.copy()
            self._commanded_gripper_width = grip_w.copy()
            self.mjdata.qpos[self.arm_qpos_indices] = arm_q
            self._apply_gripper_to_mujoco(grip_w)
            self.mjdata.qvel[self.arm_qvel_indices] = 0.0
            self.mjdata.qvel[self.gripper_qvel_indices] = 0.0
            self.mjdata.qvel[self.gripper_mirror_qvel_indices] = 0.0
            mujoco.mj_forward(self.mjmodel, self.mjdata)
            self.current_arm_qpos = arm_q.copy()
            self.current_gripper_width = grip_w.copy()
            self._safe_arm_qpos = arm_q.copy()
            self._safe_gripper_width = grip_w.copy()
            self.current_gpos = self._update_gpos_from_q(arm_q)

    def _physics_loop(self) -> None:
        interval = 1.0 / max(self.simulation_fps, 1.0)
        while not self._physics_stop.is_set():
            with self._lock:
                self._step_physics_locked()
            self._physics_stop.wait(interval)

    def _step_physics_locked(self) -> None:
        arm_q = self._commanded_arm_qpos
        grip_w = self._commanded_gripper_width
        self.mjdata.qpos[self.arm_qpos_indices] = arm_q
        self._apply_gripper_to_mujoco(grip_w)
        self.mjdata.qvel[self.arm_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_mirror_qvel_indices] = 0.0
        mujoco.mj_step(self.mjmodel, self.mjdata)
        self.mjdata.qpos[self.arm_qpos_indices] = arm_q
        self._apply_gripper_to_mujoco(grip_w)
        self.mjdata.qvel[self.arm_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_qvel_indices] = 0.0
        self.mjdata.qvel[self.gripper_mirror_qvel_indices] = 0.0
        mujoco.mj_forward(self.mjmodel, self.mjdata)

    def start(self):
        self._physics_stop.clear()
        if self._physics_thread is None:
            self._physics_thread = threading.Thread(
                target=self._physics_loop,
                name=f"{self.name}_physics_loop",
                daemon=True,
            )
            self._physics_thread.start()
        try:
            super().start()
        finally:
            self._physics_stop.set()
            if self._physics_thread is not None:
                self._physics_thread.join(timeout=1.0)
                self._physics_thread = None

    def shutdown(self):
        self._physics_stop.set()
        if self._physics_thread is not None:
            self._physics_thread.join(timeout=1.0)
            self._physics_thread = None
        super().shutdown()

    def meta2act(self, mact):
        dim = self.get_action_dim()
        return mact.get("action", np.zeros(dim))


def _test_action_writer(control_shm_name: str, mode: str, stop_event):
    from deploy.shm_utils import SharedMemoryChannel

    control_shm = SharedMemoryChannel(name=control_shm_name, max_size_mb=1, is_writer=True)
    print(f"[ActionWriter] writing {mode} actions to {control_shm_name}...")
    t = 0.0
    try:
        while not stop_event.is_set():
            if mode == "delta_ee":
                a = np.zeros(14)
                a[2] = 0.02 * np.sin(t)
                a[9] = -0.02 * np.sin(t)
                a[6] = 0.5 * np.sin(0.5 * t)
                a[13] = 0.5 * np.sin(0.5 * t)
            else:
                a = np.zeros(16)
                a[1] = 0.3 * np.sin(0.3 * t)
                a[9] = -0.3 * np.sin(0.3 * t)
                a[7] = 0.05 * (1 + np.sin(0.5 * t))
                a[15] = 0.05 * (1 + np.sin(0.5 * t))
            control_shm.write({"action": a})
            t += 0.02
            time.sleep(0.02)
    finally:
        control_shm.destroy()


def _run_robot_process(robot_config: dict):
    from deploy.base import start_device

    start_device(robot_config)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    import deploy.shm_utils  # noqa: F401

    parser = argparse.ArgumentParser(description="Test BessicaSimRobot")
    parser.add_argument("--mode", "-m", default="delta_ee", choices=["delta_ee", "qpos"])
    parser.add_argument("--visualize", "-v", action="store_true")
    args = parser.parse_args()

    robot_shm_name = "bessica_sim_test"
    control_shm_name = "bessica_sim_test_ctrl"

    from deploy.shm_utils import cleanup_all_shm

    cleanup_all_shm([robot_shm_name, control_shm_name])

    robot_config = {
        "type": "deploy.robot.bessica_sim.BessicaSimRobot",
        "args": {
            "name": robot_shm_name,
            "max_size_mb": 64,
            "fps": 50.0,
            "control_shm_name": control_shm_name,
            "control_mode": args.mode,
            "qpos_input_unit": "radians",
            "position_scale": 0.003,
            "rotation_scale": 0.02,
        },
    }

    stop_event = threading.Event()
    th = threading.Thread(target=_test_action_writer, args=(control_shm_name, args.mode, stop_event), daemon=True)
    th.start()
    time.sleep(0.5)

    proc = mp.Process(target=_run_robot_process, args=(robot_config,))
    proc.start()
    time.sleep(1.0)

    viz_proc = None
    if args.visualize:
        from deploy.visualizer.base import start_visualizer

        viz_proc = mp.Process(
            target=start_visualizer,
            args=("deploy.robot.bessica_sim.Visualizer", robot_shm_name),
            daemon=True,
        )
        viz_proc.start()

    try:
        from deploy.shm_utils import SharedMemoryChannel

        robot_shm = SharedMemoryChannel(robot_shm_name, is_writer=False, timeout=10.0)
        while proc.is_alive():
            data = robot_shm.read(blocking=False, skip_unchanged=True)
            if data is not None:
                q = data.get("qpos", np.zeros(16))
                print(f" qpos[0:3]={q[:3]}  grip={q[7]:.3f},{q[15]:.3f}", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        th.join(timeout=1.0)
        if viz_proc is not None:
            viz_proc.terminate()
            viz_proc.join(timeout=2.0)
        proc.terminate()
        proc.join(timeout=2.0)
        cleanup_all_shm([robot_shm_name, control_shm_name])
