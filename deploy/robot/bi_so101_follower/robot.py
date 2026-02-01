"""
BiSO101 Follower Robot with Camera Integration and Delta EE Control

Supports two control modes:
1. qpos: Direct joint position control (normalized space) - 12D (6 left + 6 right)
2. delta_ee: 14D delta EE control with IK (7D per arm: dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper)
"""

import numpy as np
import traceback
import time
import threading
from typing import Optional, List
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .bi_so101_follower import BiSO101FollowerConfig, BiSO101Follower
from deploy.robot.base import BaseRobot
from benchmark.base import MetaObs

# Import kinematics from so101_sim (shared implementation)
from deploy.robot.so101_sim.kinematics import create_so101, lerobot_FK, lerobot_IK


# Default control limits in radians (matching so101)
DEFAULT_QLIMIT_MIN = [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15]
DEFAULT_QLIMIT_MAX = [2.1, 0.0, 3.1, 1.475, 3.1, 1.5]

# Default gpos limits (end-effector pose limits)
DEFAULT_GLIMIT_MIN = [0.125, -0.4, 0.046, -3.1, -0.75, -1.5]
DEFAULT_GLIMIT_MAX = [0.340, 0.4, 0.23, 2.0, 1.57, 1.5]

# Default joint signs (1 = normal, -1 = reversed)
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1]


class BiSo101Follower(BaseRobot):
    """
    Bimanual SO101 Follower Robot with Camera Integration
    
    Supports two control modes:
    1. "qpos" (default): Direct joint position control in normalized space (12D)
    2. "delta_ee": 14D delta EE control with IK (7D per arm)
    """
    
    # Valid control modes
    CONTROL_MODES = ("delta_ee", "qpos")
    
    def __init__(self,
                 name: str = "bi_so101_follower",
                 max_size_mb: int = 64,
                 fps: float = 100.0,
                 control_shm_name: Optional[str] = None,
                 left_arm_port: str = "/dev/ttyACM0",
                 right_arm_port: str = "/dev/ttyACM3",
                 robot_id: str = "bi_so101_follower_arm",
                 left_arm_id: Optional[str] = None,
                 right_arm_id: Optional[str] = None,
                 camera_configs: dict = {},
                 calibration_dir: Optional[str] = None,
                 control_mode: str = "qpos",
                 position_scale: float = 0.001,
                 rotation_scale: float = 0.01,
                 gripper_scale: float = 0.01,
                 qlimit_min: Optional[List[float]] = None,
                 qlimit_max: Optional[List[float]] = None,
                 glimit_min: Optional[List[float]] = None,
                 glimit_max: Optional[List[float]] = None,
                 joint_signs: Optional[List[int]] = None,
                 left_delta_signs: Optional[List[int]] = None,
                 right_delta_signs: Optional[List[int]] = None,
                 safety_check: bool = True,
                 max_joint_delta: float = 0.15,
                 ik_tolerance: float = 0.05,
                 zero_threshold: float = 0.01,
                 debug: bool = False,
                 **kwargs):
        """
        Initialize the Bimanual SO101 Follower robot with camera
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory (for receiving actions)
            left_arm_port: Communication port for the left arm
            right_arm_port: Communication port for the right arm
            robot_id: Identifier for the robot
            left_arm_id: Custom ID for left arm (for sharing calibration)
            right_arm_id: Custom ID for right arm
            camera_configs: Dictionary of camera configurations
            calibration_dir: Optional calibration directory path
            control_mode: Control mode - "qpos" (direct joint 12D) or "delta_ee" (14D delta with IK)
            position_scale: Scale factor for position deltas (delta_ee mode)
            rotation_scale: Scale factor for rotation deltas (delta_ee mode)
            gripper_scale: Scale factor for gripper deltas (delta_ee mode)
            qlimit_min: Minimum joint limits in radians (6D, shared for both arms)
            qlimit_max: Maximum joint limits in radians (6D, shared for both arms)
            glimit_min: Minimum end-effector pose limits (6D)
            glimit_max: Maximum end-effector pose limits (6D)
            joint_signs: Joint direction signs (6D, shared for both arms)
            left_delta_signs: Delta action direction signs for left arm (7D)
            right_delta_signs: Delta action direction signs for right arm (7D)
            safety_check: Enable safety validation (FK-IK consistency, max joint delta)
            max_joint_delta: Maximum allowed joint change per step (radians)
            ik_tolerance: Maximum allowed FK-IK round-trip error (radians)
            zero_threshold: Deadzone threshold - actions smaller than this are treated as zero
            debug: Enable debug output (print qpos, gpos, actions each step)
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.left_arm_port = left_arm_port
        self.right_arm_port = right_arm_port
        self.robot_id = robot_id
        
        # Validate and set control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Action scaling (for delta_ee mode)
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        
        # Joint and end-effector limits (in radians) - shared for both arms
        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX)
        self.glimit_min = np.array(glimit_min if glimit_min is not None else DEFAULT_GLIMIT_MIN)
        self.glimit_max = np.array(glimit_max if glimit_max is not None else DEFAULT_GLIMIT_MAX)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS)
        
        # Delta action signs for each arm (for reversing input directions)
        # Format: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        self.left_delta_signs = np.array(left_delta_signs if left_delta_signs is not None else [1, 1, 1, 1, 1, 1, 1])
        self.right_delta_signs = np.array(right_delta_signs if right_delta_signs is not None else [1, 1, 1, 1, 1, 1, 1])
        
        # Kinematics (for delta_ee mode) - one for each arm
        self.robot_kin = create_so101()
        
        # Safety parameters
        self.safety_check = safety_check
        self.max_joint_delta = max_joint_delta
        self.ik_tolerance = ik_tolerance
        self._left_safety_verified = False
        self._right_safety_verified = False
        self._blocked_count = 0
        
        # Deadzone threshold for delta_ee mode
        self.zero_threshold = zero_threshold
        
        # Target state in radians (for delta_ee mode)
        # Separate state for left and right arms
        self.left_target_qpos_rad = None
        self.left_target_gpos = None
        self.right_target_qpos_rad = None
        self.right_target_gpos = None
        self._lock = threading.Lock()
        
        # Normalization offset calibration (separate for each arm)
        self._left_norm_offset = np.zeros(6, dtype=np.float64)
        self._right_norm_offset = np.zeros(6, dtype=np.float64)
        self._left_is_norm_calibrated = False
        self._right_is_norm_calibrated = False
        
        # Debug mode
        self.debug = debug
        self._debug_step = 0
        
        print(f"[BiSo101Follower] Control mode: {control_mode}")
        if control_mode == "delta_ee":
            print(f"[BiSo101Follower] Action dimension: 14D (7D left + 7D right)")
        else:
            print(f"[BiSo101Follower] Action dimension: 12D (6D left + 6D right)")
        if safety_check:
            print(f"[BiSo101Follower] Safety check ENABLED: max_joint_delta={max_joint_delta:.3f} rad, ik_tolerance={ik_tolerance:.3f} rad")
        
        # Create camera configurations
        self.camera_configs_dict = {}
        self.cameras = {}
        
        # BiSO101 arm part - using lerobot support
        robot_config = BiSO101FollowerConfig(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            id=robot_id,
            left_arm_id=left_arm_id,
            right_arm_id=right_arm_id,
            cameras=self.camera_configs_dict
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._robot = BiSO101Follower(robot_config)
        self._left_motors = list(self._robot.left_arm.bus.motors)
        self._right_motors = list(self._robot.right_arm.bus.motors)
        
        # Connect to robot
        retry_counts = 0
        max_connect_retry = 10
        while not self.connect():
            print(f"Retrying for {retry_counts} time...")
            retry_counts += 1
            if retry_counts > max_connect_retry:
                raise RuntimeError("Failed to connect to robot after max retries")
            time.sleep(1)
    
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
        """Get action dimension based on control mode"""
        if self.control_mode == "delta_ee":
            return 14  # 7D left + 7D right
        else:  # qpos
            return len(self._left_motors) + len(self._right_motors)  # 12D
    
    def _normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized joint values to radians (for a single arm)"""
        radians = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            norm_signed = normalized[i] * self.joint_signs[i]
            norm_clamped = np.clip(norm_signed, -100.0, 100.0)
            radians[i] = (norm_clamped + 100.0) / 200.0 * (qmax - qmin) + qmin
        
        # Joint 5 (gripper): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        norm_signed = normalized[5] * self.joint_signs[5]
        norm_clamped = np.clip(norm_signed, 0.0, 100.0)
        radians[5] = norm_clamped / 100.0 * (qmax - qmin) + qmin
        
        return radians
    
    def _radians_to_normalized(self, radians: np.ndarray, norm_offset: np.ndarray, is_calibrated: bool) -> np.ndarray:
        """Convert radians to normalized joint values (for a single arm)"""
        normalized = np.zeros(6, dtype=np.float64)
        for i in range(5):  # Joints 0-4: RANGE_M100_100 [-100, 100]
            qmin = self.qlimit_min[i]
            qmax = self.qlimit_max[i]
            rad_clamped = np.clip(radians[i], qmin, qmax)
            normalized[i] = ((rad_clamped - qmin) / (qmax - qmin)) * 200.0 - 100.0
            normalized[i] *= self.joint_signs[i]
        
        # Joint 5 (gripper): RANGE_0_100 [0, 100]
        qmin = self.qlimit_min[5]
        qmax = self.qlimit_max[5]
        rad_clamped = np.clip(radians[5], qmin, qmax)
        normalized[5] = ((rad_clamped - qmin) / (qmax - qmin)) * 100.0
        normalized[5] *= self.joint_signs[5]
        
        # Apply calibration offset if calibrated
        if is_calibrated:
            normalized = normalized + norm_offset
        
        return normalized
    
    def _verify_fk_ik_consistency(self, qpos_rad: np.ndarray, arm_name: str) -> bool:
        """Safety check: Verify FK-IK round-trip consistency for one arm"""
        try:
            gpos = lerobot_FK(qpos_rad[1:5], robot=self.robot_kin)
            qpos_recovered, ik_success = lerobot_IK(qpos_rad[1:5], gpos, robot=self.robot_kin)
            
            if not ik_success:
                print(f"[SAFETY-{arm_name}] FK-IK verification FAILED: IK did not converge")
                return False
            
            error = np.abs(qpos_rad[1:5] - qpos_recovered[:4])
            max_error = np.max(error)
            
            if max_error > self.ik_tolerance:
                print(f"[SAFETY-{arm_name}] FK-IK verification FAILED: max_error={max_error:.4f} rad > tolerance={self.ik_tolerance:.4f} rad")
                return False
            
            print(f"[SAFETY-{arm_name}] FK-IK verification PASSED: max_error={max_error:.6f} rad")
            return True
            
        except Exception as e:
            print(f"[SAFETY-{arm_name}] FK-IK verification FAILED with exception: {e}")
            return False
    
    def _apply_deadzone(self, action: np.ndarray) -> np.ndarray:
        """Apply deadzone threshold to action"""
        result = action.copy()
        for i in range(len(result)):
            if abs(result[i]) < self.zero_threshold:
                result[i] = 0.0
        return result
    
    def _calibrate_normalization(self, qpos_normalized: np.ndarray, qpos_rad: np.ndarray, 
                                  arm_name: str, norm_offset: np.ndarray, is_calibrated: bool) -> tuple:
        """Calibrate normalization offset for one arm"""
        qpos_normalized_recovered = self._radians_to_normalized(qpos_rad, np.zeros(6), False)
        offset = qpos_normalized - qpos_normalized_recovered
        
        print(f"\n[CALIBRATION-{arm_name}] Normalization calibration complete!")
        print(f"              Original normalized:  {qpos_normalized}")
        print(f"              Recovered normalized: {qpos_normalized_recovered}")
        print(f"              Offset:               {offset}")
        
        max_offset = np.max(np.abs(offset))
        if max_offset > 1.0:
            print(f"              WARNING: Large offset detected ({max_offset:.2f})!")
        else:
            print(f"              Offset is small (max={max_offset:.4f}), calibration looks good.")
        
        return offset, True
    
    def get_observation(self):
        """Get complete observation data (including camera images)"""
        try:
            obs = self._robot.get_observation()
            left_qpos = np.array([obs['left_'+m+'.pos'] for m in self._left_motors], dtype=np.float64)
            right_qpos = np.array([obs['right_'+m+'.pos'] for m in self._right_motors], dtype=np.float64)
            
            # Update internal state for delta_ee mode
            if self.control_mode == "delta_ee":
                left_qpos_rad = self._normalized_to_radians(left_qpos)
                right_qpos_rad = self._normalized_to_radians(right_qpos)
                
                # Calibration for each arm
                if not self._left_is_norm_calibrated:
                    self._left_norm_offset, self._left_is_norm_calibrated = self._calibrate_normalization(
                        left_qpos, left_qpos_rad, "LEFT", self._left_norm_offset, self._left_is_norm_calibrated)
                
                if not self._right_is_norm_calibrated:
                    self._right_norm_offset, self._right_is_norm_calibrated = self._calibrate_normalization(
                        right_qpos, right_qpos_rad, "RIGHT", self._right_norm_offset, self._right_is_norm_calibrated)
                
                # First-time initialization of target state
                if self.left_target_qpos_rad is None:
                    print("\n" + "="*60)
                    print("[INIT-LEFT] Initializing left arm target state...")
                    print(f"            qpos (normalized): {left_qpos}")
                    print(f"            qpos (radians):    {left_qpos_rad}")
                    
                    with self._lock:
                        self.left_target_qpos_rad = left_qpos_rad.copy()
                        self.left_target_gpos = lerobot_FK(left_qpos_rad[1:5], robot=self.robot_kin)
                    
                    print(f"            Initial gpos:      {self.left_target_gpos}")
                    
                    if self.safety_check:
                        if self._verify_fk_ik_consistency(left_qpos_rad, "LEFT"):
                            self._left_safety_verified = True
                        else:
                            print("[SAFETY-LEFT] WARNING: FK-IK consistency check FAILED!")
                    else:
                        self._left_safety_verified = True
                    print("="*60 + "\n")
                
                if self.right_target_qpos_rad is None:
                    print("\n" + "="*60)
                    print("[INIT-RIGHT] Initializing right arm target state...")
                    print(f"             qpos (normalized): {right_qpos}")
                    print(f"             qpos (radians):    {right_qpos_rad}")
                    
                    with self._lock:
                        self.right_target_qpos_rad = right_qpos_rad.copy()
                        self.right_target_gpos = lerobot_FK(right_qpos_rad[1:5], robot=self.robot_kin)
                    
                    print(f"             Initial gpos:      {self.right_target_gpos}")
                    
                    if self.safety_check:
                        if self._verify_fk_ik_consistency(right_qpos_rad, "RIGHT"):
                            self._right_safety_verified = True
                        else:
                            print("[SAFETY-RIGHT] WARNING: FK-IK consistency check FAILED!")
                    else:
                        self._right_safety_verified = True
                    print("="*60 + "\n")
            
            return {'qpos': np.concatenate([left_qpos, right_qpos]), **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None
    
    def _process_arm_delta_action(self, action_7d: np.ndarray, delta_signs: np.ndarray,
                                   target_qpos_rad: np.ndarray, target_gpos: np.ndarray,
                                   norm_offset: np.ndarray, is_calibrated: bool,
                                   arm_name: str) -> tuple:
        """
        Process 7D delta action for one arm and return target normalized qpos
        
        Returns: (target_qpos_normalized, new_target_qpos_rad, new_target_gpos, success)
        """
        # Apply delta signs and scale
        action_signed = action_7d * delta_signs
        delta_pos = action_signed[0:3] * self.position_scale
        delta_rot = action_signed[3:6] * self.rotation_scale
        delta_gripper = action_signed[6] * self.gripper_scale
        
        current_gpos = target_gpos.copy()
        current_qpos_rad = target_qpos_rad.copy()
        
        # Check if all deltas are zero
        total_delta = np.sum(np.abs(delta_pos)) + np.sum(np.abs(delta_rot)) + np.abs(delta_gripper)
        if total_delta < 1e-10:
            return None, target_qpos_rad, target_gpos, True  # No change needed
        
        # Compute new target gpos (world-frame XY motion)
        angle_curr = current_qpos_rad[0]  # Base rotation
        forward_curr = current_gpos[0]    # Forward distance
        
        x_curr = forward_curr * np.cos(angle_curr)
        y_curr = forward_curr * np.sin(angle_curr)
        
        x_new = x_curr + delta_pos[0]
        y_new = y_curr + delta_pos[1]
        
        theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)
        
        new_target_gpos = current_gpos.copy()
        new_target_gpos[0] += forward_update  # Forward
        new_target_gpos[2] += delta_pos[2]    # Height (Z)
        new_target_gpos[3] += delta_rot[0]    # Roll
        new_target_gpos[4] += delta_rot[1]    # Pitch
        
        # Compute IK for joints 1-4
        fd_qpos = current_qpos_rad[1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, new_target_gpos, robot=self.robot_kin)
        
        if not ik_success:
            # Try reducing delta
            for scale in [0.5, 0.25, 0.1]:
                scaled_gpos = current_gpos.copy()
                scaled_gpos[0] += forward_update * scale
                scaled_gpos[2] += delta_pos[2] * scale
                scaled_gpos[3] += delta_rot[0] * scale
                scaled_gpos[4] += delta_rot[1] * scale
                
                qpos_inv, ik_success = lerobot_IK(fd_qpos, scaled_gpos, robot=self.robot_kin)
                if ik_success:
                    new_target_gpos = scaled_gpos
                    theta_update *= scale
                    break
            
            if not ik_success:
                if self.debug:
                    print(f"[BiSo101Follower-{arm_name}] IK failed for target_gpos={new_target_gpos}")
                return None, target_qpos_rad, target_gpos, False
        
        # Build target joint positions
        new_target_qpos_rad = current_qpos_rad.copy()
        new_target_qpos_rad[0] += theta_update
        new_target_qpos_rad[1:5] = qpos_inv[:4]
        new_target_qpos_rad[5] += delta_gripper
        
        # Convert to normalized space
        target_qpos_normalized = self._radians_to_normalized(new_target_qpos_rad, norm_offset, is_calibrated)
        
        # Clip in normalized space
        target_qpos_normalized[:5] = np.clip(target_qpos_normalized[:5], -100.0, 100.0)
        target_qpos_normalized[5] = np.clip(target_qpos_normalized[5], 0.0, 100.0)
        
        return target_qpos_normalized, new_target_qpos_rad, new_target_gpos, True
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode
        
        For qpos mode: pass through (12D normalized values: 6 left + 6 right)
        For delta_ee mode: convert 14D delta to 12D normalized joint positions
        """
        if self.control_mode == "qpos":
            return action_dict
        
        # delta_ee mode
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        # Apply deadzone threshold
        if self.zero_threshold > 0:
            action = self._apply_deadzone(action)
        
        # Safety check: Block if FK-IK not verified for either arm
        if self.safety_check and (not self._left_safety_verified or not self._right_safety_verified):
            self._blocked_count += 1
            if self._blocked_count % 100 == 1:
                print(f"[SAFETY] Action BLOCKED (#{self._blocked_count}): FK-IK consistency not verified. "
                      f"Left verified: {self._left_safety_verified}, Right verified: {self._right_safety_verified}")
            action_dict['action'] = None
            return action_dict
        
        if len(action) != 14:
            print(f"[BiSo101Follower] Warning: delta_ee mode expects 14D action, got {len(action)}D")
            return action_dict
        
        # Split action for left and right arms
        left_action = action[:7]
        right_action = action[7:]
        
        with self._lock:
            if self.left_target_qpos_rad is None or self.right_target_qpos_rad is None:
                return action_dict
            
            left_target_qpos_rad = self.left_target_qpos_rad.copy()
            left_target_gpos = self.left_target_gpos.copy()
            right_target_qpos_rad = self.right_target_qpos_rad.copy()
            right_target_gpos = self.right_target_gpos.copy()
        
        # Process left arm
        left_qpos_normalized, new_left_qpos_rad, new_left_gpos, left_success = self._process_arm_delta_action(
            left_action, self.left_delta_signs, left_target_qpos_rad, left_target_gpos,
            self._left_norm_offset, self._left_is_norm_calibrated, "LEFT"
        )
        
        # Process right arm
        right_qpos_normalized, new_right_qpos_rad, new_right_gpos, right_success = self._process_arm_delta_action(
            right_action, self.right_delta_signs, right_target_qpos_rad, right_target_gpos,
            self._right_norm_offset, self._right_is_norm_calibrated, "RIGHT"
        )
        
        # Update internal target state
        with self._lock:
            if left_success and left_qpos_normalized is not None:
                self.left_target_qpos_rad = new_left_qpos_rad.copy()
                self.left_target_gpos = new_left_gpos.copy()
            if right_success and right_qpos_normalized is not None:
                self.right_target_qpos_rad = new_right_qpos_rad.copy()
                self.right_target_gpos = new_right_gpos.copy()
        
        # Combine actions for both arms
        if left_qpos_normalized is None and right_qpos_normalized is None:
            # Both arms have no change
            action_dict['action'] = None
        else:
            # Use current normalized position for arms with no change
            if left_qpos_normalized is None:
                left_qpos_normalized = self._radians_to_normalized(
                    left_target_qpos_rad, self._left_norm_offset, self._left_is_norm_calibrated)
            if right_qpos_normalized is None:
                right_qpos_normalized = self._radians_to_normalized(
                    right_target_qpos_rad, self._right_norm_offset, self._right_is_norm_calibrated)
            
            action_dict['action'] = np.concatenate([left_qpos_normalized, right_qpos_normalized])
        
        # Debug output
        if self.debug and action_dict.get('action') is not None:
            self._debug_step += 1
            if self._debug_step % 10 == 1:
                print(f"\n[DEBUG step {self._debug_step}] ================")
                print(f"  Input action (14D): {action}")
                print(f"  Left arm action:    {left_action}")
                print(f"  Right arm action:   {right_action}")
                print(f"  ---")
                print(f"  Left target gpos:   {new_left_gpos}")
                print(f"  Right target gpos:  {new_right_gpos}")
                if action_dict.get('action') is not None:
                    output = action_dict['action']
                    print(f"  Output (12D norm):  L={output[:6]}, R={output[6:]}")
        
        return action_dict
    
    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        if obs is None:
            return None
        
        left_qpos = np.array([obs['left_'+mname+'.pos'] for mname in self._left_motors], dtype=np.float32)
        right_qpos = np.array([obs['right_'+mname+'.pos'] for mname in self._right_motors], dtype=np.float32)
        qpos = np.concatenate([left_qpos, right_qpos])
        
        if 'front_camera' in obs:
            image = obs['front_camera'][np.newaxis, :].transpose(0, 3, 1, 2)
        else:
            image = np.zeros((1, 3, 480, 640), dtype=np.uint8)
            
        return MetaObs(state=qpos, state_joint=qpos, image=image)
    
    def shutdown(self):
        """Shutdown robot and cameras"""
        if self._robot.is_connected:
            self._robot.disconnect()
    
    def close(self):
        """Close robot"""
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()
    
    def publish_action(self, action: np.ndarray):
        """Publish action to robot (12D: 6 left + 6 right)"""
        try:
            left_action = {'left_'+mname+'.pos': action[i] for i, mname in enumerate(self._left_motors)}
            right_action = {'right_'+mname+'.pos': action[i+len(self._left_motors)] for i, mname in enumerate(self._right_motors)}
            action_dict = {**left_action, **right_action}
            self._robot.send_action(action_dict)
        except Exception as e:
            pass
    
    def is_running(self):
        """Check if robot is running"""
        return self._robot.is_connected
    
    def save_episode(self, file_path: str, observations: list, actions: list):
        """Save episode data to HDF5 file"""
        import h5py
        import os
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        def write_group(group, data_list, key_prefix=None):
            if isinstance(data_list[0], dict):
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


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import importlib
    import threading
    import yaml
    from pathlib import Path

    from deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "robot" / "bi_so101_follower.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    device_config = raw[0] if isinstance(raw, list) else raw

    # Create device in main process
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])

    shm_name = device_config["args"].get("name", device_config["args"]["robot_id"])
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from BiSo101 Follower SHM (Ctrl+C to stop)...")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=15.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None:
                arr = data.get("qpos")
                if arr is not None:
                    n = len(arr)
                    if n >= 12:
                        left, right = arr[:6], arr[6:12]
                        print(
                            f"  left: [{left[0]:.3f}, {left[1]:.3f}, {left[2]:.3f}, "
                            f"{left[3]:.3f}, {left[4]:.3f}, {left[5]:.3f}]  "
                            f"right: [{right[0]:.3f}, {right[1]:.3f}, {right[2]:.3f}, "
                            f"{right[3]:.3f}, {right[4]:.3f}, {right[5]:.3f}]",
                            end="\r",
                            flush=True,
                        )
                    else:
                        print(f"  qpos: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")
