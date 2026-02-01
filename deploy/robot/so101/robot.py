"""
SO101 Follower Robot with Camera Integration
Integrates camera directly into lerobot's SO101Follower

Supports two control modes:
1. qpos: Direct joint position control (normalized space)
2. delta_ee: 7D delta EE control with IK (dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper)
"""

import numpy as np
import traceback
import time
import threading
from typing import Optional, List
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
try:
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
except ImportError:
    from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from deploy.robot.base import BaseRobot
from benchmark.base import MetaObs

# Import kinematics from so101_sim (shared implementation)
from deploy.robot.so101_sim.kinematics import create_so101, lerobot_FK, lerobot_IK


# Default control limits in radians (matching so101_sim)
DEFAULT_QLIMIT_MIN = [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15]
DEFAULT_QLIMIT_MAX = [2.1, 0.0, 3.1, 1.475, 3.1, 1.5]

# Default gpos limits (end-effector pose limits)
DEFAULT_GLIMIT_MIN = [0.125, -0.4, 0.046, -3.1, -0.75, -1.5]
DEFAULT_GLIMIT_MAX = [0.340, 0.4, 0.23, 2.0, 1.57, 1.5]

# Default joint signs (1 = normal, -1 = reversed)
DEFAULT_JOINT_SIGNS = [1, 1, 1, 1, 1, 1]


class So101FollowerWithCamera(BaseRobot):
    """
    Integrate camera directly into So101Follower
    So that the camera becomes part of the robot, rather than an external component
    
    Supports two control modes:
    1. "qpos" (default): Direct joint position control in normalized space
    2. "delta_ee": 7D delta EE control with IK
    """
    
    # Valid control modes
    CONTROL_MODES = ("delta_ee", "qpos")
    
    def __init__(self,
                 name: str = "so101_follower",
                 max_size_mb: int = 64,
                 fps: float = 100.0,
                 control_shm_name: Optional[str] = None,
                 com: str = "/dev/ttyACM0",
                 robot_id: str = "so101_follower_arm",
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
                 delta_signs: Optional[List[int]] = None,
                 safety_check: bool = True,
                 max_joint_delta: float = 0.15,
                 ik_tolerance: float = 0.05,
                 zero_threshold: float = 0.01,
                 debug: bool = False,
                 **kwargs):
        """
        Initialize the SO101 Follower robot with camera
        
        Args:
            name: Name for the shared memory segment
            max_size_mb: Maximum size of shared memory in MB
            fps: Control frequency in Hz
            control_shm_name: Name of the control shared memory (for receiving actions)
            com: Communication port for the robot
            robot_id: Identifier for the robot
            camera_configs: Dictionary of camera configurations
            calibration_dir: Optional calibration directory path
            control_mode: Control mode - "qpos" (direct joint) or "delta_ee" (7D delta with IK)
            position_scale: Scale factor for position deltas (delta_ee mode)
            rotation_scale: Scale factor for rotation deltas (delta_ee mode)
            gripper_scale: Scale factor for gripper deltas (delta_ee mode)
            qlimit_min: Minimum joint limits in radians (6D)
            qlimit_max: Maximum joint limits in radians (6D)
            glimit_min: Minimum end-effector pose limits (6D)
            glimit_max: Maximum end-effector pose limits (6D)
            joint_signs: Joint direction signs (6D, 1=normal, -1=reversed)
            delta_signs: Delta action direction signs (7D: dx, dy, dz, droll, dpitch, dyaw, dgripper)
            safety_check: Enable safety validation (FK-IK consistency, max joint delta)
            max_joint_delta: Maximum allowed joint change per step (radians)
            ik_tolerance: Maximum allowed FK-IK round-trip error (radians)
            zero_threshold: Deadzone threshold - actions smaller than this are treated as zero
            debug: Enable debug output (print qpos, gpos, actions each step)
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.com = com
        self.robot_id = robot_id
        
        # Validate and set control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Action scaling (for delta_ee mode)
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        
        # Joint and end-effector limits (in radians)
        self.qlimit_min = np.array(qlimit_min if qlimit_min is not None else DEFAULT_QLIMIT_MIN)
        self.qlimit_max = np.array(qlimit_max if qlimit_max is not None else DEFAULT_QLIMIT_MAX)
        self.glimit_min = np.array(glimit_min if glimit_min is not None else DEFAULT_GLIMIT_MIN)
        self.glimit_max = np.array(glimit_max if glimit_max is not None else DEFAULT_GLIMIT_MAX)
        self.joint_signs = np.array(joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS)
        
        # Delta action signs (for reversing input directions)
        # Format: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        self.delta_signs = np.array(delta_signs if delta_signs is not None else [1, 1, 1, 1, 1, 1, 1])
        
        # Kinematics (for delta_ee mode)
        self.robot_kin = create_so101()
        
        # Safety parameters
        self.safety_check = safety_check
        self.max_joint_delta = max_joint_delta  # Max radians change per step
        self.ik_tolerance = ik_tolerance        # Max FK-IK round-trip error
        self._safety_verified = False           # Whether FK-IK consistency is verified
        self._blocked_count = 0                 # Count of blocked unsafe actions
        
        # Deadzone threshold for delta_ee mode
        self.zero_threshold = zero_threshold
        
        # Target state in radians (for delta_ee mode)
        # IMPORTANT: We maintain internal target state, not reading from robot feedback
        # This avoids IK drift caused by sensor noise and normalization errors
        self.target_qpos_rad = None  # Will be initialized on first observation
        self.target_gpos = None      # Target end-effector pose (used for IK input)
        self._lock = threading.Lock()
        
        # Normalization offset calibration
        # This compensates for the difference between theoretical and actual normalized↔radians mapping
        self._norm_offset = np.zeros(6, dtype=np.float64)  # Offset to add to normalized output
        self._is_norm_calibrated = False
        
        # Debug mode
        self.debug = debug
        self._debug_step = 0
        
        print(f"[So101FollowerWithCamera] Control mode: {control_mode}")
        if safety_check:
            print(f"[So101FollowerWithCamera] Safety check ENABLED: max_joint_delta={max_joint_delta:.3f} rad, ik_tolerance={ik_tolerance:.3f} rad")
        
        # Create camera configurations
        camera_configs_dict = {}
        
        # SO101 arm part - using official lerobot support
        robot_config = SO101FollowerConfig(
            port=com,
            id=robot_id,
            cameras=camera_configs_dict
        )
        if calibration_dir:
            robot_config.calibration_dir = Path(calibration_dir)
        self._robot = SO101Follower(robot_config)
        self._motors = list(self._robot.bus.motors)
        
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
            return 7  # 6-DOF pose delta + gripper
        else:  # qpos
            return len(self._motors)  # 6 joint positions
    
    def _normalized_to_radians(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized joint values to radians
        
        SO101 uses different normalization for different joints:
        - Joints 0-4 (body): MotorNormMode.RANGE_M100_100, range [-100, 100]
        - Joint 5 (gripper): MotorNormMode.RANGE_0_100, range [0, 100]
        
        Args:
            normalized: 6D array with normalized values
            
        Returns:
            6D array with values in radians
        """
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
    
    def _radians_to_normalized(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert radians to normalized joint values
        
        SO101 uses different normalization for different joints:
        - Joints 0-4 (body): MotorNormMode.RANGE_M100_100, range [-100, 100]
        - Joint 5 (gripper): MotorNormMode.RANGE_0_100, range [0, 100]
        
        After calibration, applies the calibrated offset to ensure round-trip consistency.
        
        Args:
            radians: 6D array with values in radians
            
        Returns:
            6D array with normalized values
        """
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
        if self._is_norm_calibrated:
            normalized = normalized + self._norm_offset
        
        return normalized
    
    def _verify_fk_ik_consistency(self, qpos_rad: np.ndarray) -> bool:
        """
        Safety check 1: Verify FK-IK round-trip consistency
        
        Computes: qpos -> FK -> gpos -> IK -> qpos_recovered
        Returns True if |qpos - qpos_recovered| < ik_tolerance for joints 1-4
        
        This ensures the kinematics are working correctly before trusting IK results.
        """
        try:
            # FK: qpos[1:5] -> gpos
            gpos = lerobot_FK(qpos_rad[1:5], robot=self.robot_kin)
            
            # IK: gpos -> qpos_recovered[1:5]
            qpos_recovered, ik_success = lerobot_IK(qpos_rad[1:5], gpos, robot=self.robot_kin)
            
            if not ik_success:
                print(f"[SAFETY] FK-IK verification FAILED: IK did not converge")
                return False
            
            # Compare joints 1-4 (the ones used in FK/IK)
            error = np.abs(qpos_rad[1:5] - qpos_recovered[:4])
            max_error = np.max(error)
            
            if max_error > self.ik_tolerance:
                print(f"[SAFETY] FK-IK verification FAILED: max_error={max_error:.4f} rad > tolerance={self.ik_tolerance:.4f} rad")
                print(f"         qpos_orig[1:5]:      {qpos_rad[1:5]}")
                print(f"         qpos_recovered[1:5]: {qpos_recovered[:4]}")
                print(f"         error:               {error}")
                return False
            
            print(f"[SAFETY] FK-IK verification PASSED: max_error={max_error:.6f} rad")
            return True
            
        except Exception as e:
            print(f"[SAFETY] FK-IK verification FAILED with exception: {e}")
            return False
    
    def _check_joint_delta_safe(self, current_qpos_rad: np.ndarray, target_qpos_rad: np.ndarray) -> tuple:
        """
        Safety check 2: Verify target qpos is not too far from current qpos
        
        Returns: (is_safe, max_delta, problematic_joint_idx)
        """
        delta = np.abs(target_qpos_rad - current_qpos_rad)
        max_delta = np.max(delta)
        max_joint_idx = np.argmax(delta)
        
        if max_delta > self.max_joint_delta:
            return False, max_delta, max_joint_idx
        return True, max_delta, max_joint_idx
    
    def _apply_deadzone(self, action: np.ndarray) -> np.ndarray:
        """
        Apply deadzone threshold to action
        
        Args:
            action: Raw 7D delta action
            
        Returns:
            Action with deadzone applied
        """
        result = action.copy()
        for i in range(len(result)):
            if abs(result[i]) < self.zero_threshold:
                result[i] = 0.0
        return result

    def get_observation(self):
        """Get complete observation data (including camera images)"""
        try:
            obs = self._robot.get_observation()
            qpos_normalized = np.array([obs[mname + '.pos'] for mname in self._motors], dtype=np.float64)
            
            # Update internal state for delta_ee mode
            if self.control_mode == "delta_ee":
                qpos_rad = self._normalized_to_radians(qpos_normalized)
                
                # First-time normalization calibration:
                # Compute offset so that radians_to_normalized(normalized_to_radians(n)) == n
                if not self._is_norm_calibrated:
                    self._calibrate_normalization(qpos_normalized, qpos_rad)
                
                # First-time initialization of target state
                if self.target_qpos_rad is None:
                    print("\n" + "="*60)
                    print("[INIT] Initializing target state from current robot position...")
                    print(f"       Current qpos (normalized): {qpos_normalized}")
                    print(f"       Current qpos (radians):    {qpos_rad}")
                    
                    with self._lock:
                        self.target_qpos_rad = qpos_rad.copy()
                        self.target_gpos = lerobot_FK(qpos_rad[1:5], robot=self.robot_kin)
                    
                    print(f"       Initial gpos:              {self.target_gpos}")
                    
                    # First-time safety verification: check FK-IK consistency
                    if self.safety_check:
                        print("[SAFETY] Verifying FK-IK consistency...")
                        if self._verify_fk_ik_consistency(qpos_rad):
                            self._safety_verified = True
                            print("[SAFETY] FK-IK consistency verified. Safe to proceed.")
                        else:
                            print("[SAFETY] WARNING: FK-IK consistency check FAILED!")
                            print("[SAFETY] Delta EE actions will be BLOCKED until consistency is verified.")
                    else:
                        self._safety_verified = True
                    
                    print("="*60 + "\n")
            
            return {'qpos': qpos_normalized, **obs}
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None
    
    def _calibrate_normalization(self, qpos_normalized: np.ndarray, qpos_rad: np.ndarray):
        """
        Calibrate the normalization offset to ensure round-trip consistency.
        
        The issue is that our theoretical qlimit_min/max might not match the actual
        hardware calibration range_min/range_max. This causes drift when we:
        1. Read normalized value n
        2. Convert to radians: r = normalized_to_radians(n)
        3. Convert back: n' = radians_to_normalized(r)
        4. If n' != n, there's drift!
        
        Solution: Compute offset = n - n' and add it to all normalized outputs.
        """
        # Round-trip: normalized -> radians -> normalized
        qpos_normalized_recovered = self._radians_to_normalized(qpos_rad)
        
        # Compute offset: what we need to add to match original
        self._norm_offset = qpos_normalized - qpos_normalized_recovered
        self._is_norm_calibrated = True
        
        print("\n" + "="*60)
        print("[CALIBRATION] Normalization calibration complete!")
        print(f"              Original normalized:  {qpos_normalized}")
        print(f"              Recovered normalized: {qpos_normalized_recovered}")
        print(f"              Offset:               {self._norm_offset}")
        
        # Check if offset is significant
        max_offset = np.max(np.abs(self._norm_offset))
        if max_offset > 1.0:
            print(f"              WARNING: Large offset detected ({max_offset:.2f})!")
            print("              This may indicate qlimit_min/max don't match hardware calibration.")
        else:
            print(f"              Offset is small (max={max_offset:.4f}), calibration looks good.")
        print("="*60 + "\n")
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode
        
        For qpos mode: pass through (normalized values)
        For delta_ee mode: convert 7D delta to 6D normalized joint positions
        
        With safety_check enabled:
        1. Block all actions until FK-IK consistency is verified
        2. Block actions that would cause large joint deltas
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
        
        # Safety check 1: Block if FK-IK not verified
        if self.safety_check and not self._safety_verified:
            self._blocked_count += 1
            if self._blocked_count % 100 == 1:  # Print every 100 blocked actions
                print(f"[SAFETY] Action BLOCKED (#{self._blocked_count}): FK-IK consistency not verified. "
                      f"Move robot to a known good position and restart.")
            # Return empty action to prevent robot movement
            action_dict['action'] = None
            return action_dict
        
        if len(action) != 7:
            print(f"[So101FollowerWithCamera] Warning: delta_ee mode expects 7D action, got {len(action)}D")
            return action_dict
        
        # Apply delta signs (for reversing input directions) and scale
        action_signed = action * self.delta_signs
        delta_pos = action_signed[0:3] * self.position_scale
        delta_rot = action_signed[3:6] * self.rotation_scale
        delta_gripper = action_signed[6] * self.gripper_scale
        
        with self._lock:
            if self.target_qpos_rad is None or self.target_gpos is None:
                # Not initialized yet, skip
                return action_dict
            
            # IMPORTANT: Use internal target state, not robot feedback!
            # This avoids IK drift caused by sensor noise and normalization errors
            current_gpos = self.target_gpos.copy()
            current_qpos_rad = self.target_qpos_rad.copy()
        
        # CRITICAL: If all deltas are zero, return None to keep current position
        # This avoids unnecessary IK computation
        total_delta = np.sum(np.abs(delta_pos)) + np.sum(np.abs(delta_rot)) + np.abs(delta_gripper)
        if total_delta < 1e-10:
            action_dict['action'] = None
            return action_dict
        
        # Compute new target gpos (same logic as so101_sim and real_keyboard_control.py)
        # Handle XY motion in world frame
        angle_curr = current_qpos_rad[0]  # Base rotation
        forward_curr = current_gpos[0]    # Forward distance
        
        # Current XY in world frame
        x_curr = forward_curr * np.cos(angle_curr)
        y_curr = forward_curr * np.sin(angle_curr)
        
        # Apply delta in world frame
        x_new = x_curr + delta_pos[0]
        y_new = y_curr + delta_pos[1]
        
        # Convert back to polar coordinates
        theta_update = np.arctan2(y_new, x_new) - np.arctan2(y_curr, x_curr)
        forward_update = np.sqrt(x_new**2 + y_new**2) - np.sqrt(x_curr**2 + y_curr**2)
        
        # Update target gpos (accumulative, not from sensor feedback)
        target_gpos = current_gpos.copy()
        target_gpos[0] += forward_update  # Forward
        target_gpos[2] += delta_pos[2]    # Height (Z)
        target_gpos[3] += delta_rot[0]    # Roll
        target_gpos[4] += delta_rot[1]    # Pitch
        
        # Compute IK for joints 1-4 (Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
        fd_qpos = current_qpos_rad[1:5]
        qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=self.robot_kin)
        
        if not ik_success:
            # IK failed - try reducing the delta (binary search for feasible motion)
            # This helps when the full delta is unreachable but partial motion is possible
            for scale in [0.5, 0.25, 0.1]:
                scaled_gpos = current_gpos.copy()
                scaled_gpos[0] += forward_update * scale
                scaled_gpos[2] += delta_pos[2] * scale
                scaled_gpos[3] += delta_rot[0] * scale
                scaled_gpos[4] += delta_rot[1] * scale
                
                qpos_inv, ik_success = lerobot_IK(fd_qpos, scaled_gpos, robot=self.robot_kin)
                if ik_success:
                    target_gpos = scaled_gpos
                    theta_update *= scale  # Also scale base rotation
                    break
            
            if not ik_success:
                # Still failed, keep current position
                if self.debug:
                    print(f"[So101FollowerWithCamera] IK failed for target_gpos={target_gpos}")
                action_dict['action'] = None
                return action_dict
        
        # Build target joint positions
        target_qpos_rad = current_qpos_rad.copy()
        target_qpos_rad[0] += theta_update           # Base rotation
        target_qpos_rad[1:5] = qpos_inv[:4]          # IK solution for arm joints
        target_qpos_rad[5] += delta_gripper          # Gripper
        
        # Convert from radians to normalized space for robot
        target_qpos_normalized = self._radians_to_normalized(target_qpos_rad)
        
        # Clip in normalized space (the actual robot's native space)
        # Joints 0-4: RANGE_M100_100 [-100, 100]
        # Joint 5 (gripper): RANGE_0_100 [0, 100]
        target_qpos_normalized[:5] = np.clip(target_qpos_normalized[:5], -100.0, 100.0)
        target_qpos_normalized[5] = np.clip(target_qpos_normalized[5], 0.0, 100.0)
        
        # Update internal target state
        with self._lock:
            self.target_qpos_rad = target_qpos_rad.copy()
            self.target_gpos = target_gpos.copy()
        
        # Debug output
        if self.debug:
            self._debug_step += 1
            if self._debug_step % 10 == 1:  # Print every 10 steps
                # Check which joints are at their normalized limits
                limit_info = []
                joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
                for i in range(5):  # Joints 0-4: [-100, 100]
                    if target_qpos_normalized[i] <= -100.0 + 0.1:
                        limit_info.append(f"{joint_names[i]}@-100")
                    elif target_qpos_normalized[i] >= 100.0 - 0.1:
                        limit_info.append(f"{joint_names[i]}@+100")
                # Joint 5 (gripper): [0, 100]
                if target_qpos_normalized[5] <= 0.1:
                    limit_info.append(f"{joint_names[5]}@0")
                elif target_qpos_normalized[5] >= 99.9:
                    limit_info.append(f"{joint_names[5]}@100")
                
                print(f"\n[DEBUG step {self._debug_step}] ================")
                print(f"  Input action (raw):      {action}")
                print(f"  Delta pos (scaled):      {delta_pos}")
                print(f"  Delta rot (scaled):      {delta_rot}")
                print(f"  Delta gripper (scaled):  {delta_gripper:.6f}")
                print(f"  ---")
                print(f"  Prev target qpos (rad):  {current_qpos_rad}")
                print(f"  Prev target gpos:        {current_gpos}")
                print(f"  ---")
                print(f"  New target qpos (rad):   {target_qpos_rad}")
                print(f"  New target gpos:         {target_gpos}")
                print(f"  New target qpos (norm):  {target_qpos_normalized}")
                print(f"  ---")
                print(f"  Qpos delta (rad):        {target_qpos_rad - current_qpos_rad}")
                if limit_info:
                    print(f"  [LIMIT] Joint at limit:  {', '.join(limit_info)}")
        
        action_dict['action'] = target_qpos_normalized
        return action_dict

    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        if obs is None:
            return None
        
        qpos = obs.get('qpos')
        if qpos is None:
            qpos = np.array([obs[mname + '.pos'] for mname in self._motors], dtype=np.float32)
        
        # Process image data
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
        """Close the robot"""
        super().close()
        if self._robot.is_connected:
            self._robot.disconnect()
        
    def publish_action(self, action: np.ndarray):
        """Publish action to robot"""
        try:
            action_dict = {mname + '.pos': action[i] for i, mname in enumerate(self._motors)}
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
# Test (run Follower in main process so connect/retry works, then read SHM)
# ==============================================================================

if __name__ == "__main__":
    import importlib
    import threading
    import yaml
    from pathlib import Path

    from deploy.shm_utils import SharedMemoryChannel

    # Load config
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "robot" / "so101_follower.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    device_config = raw[0] if isinstance(raw, list) else raw

    # Create device in main process
    device_type = device_config["type"]
    module_name, class_name = device_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    device_class = getattr(module, class_name)
    device = device_class(**device_config["args"])

    shm_name = device_config["args"]["name"]
    start_thread = threading.Thread(target=device.start, daemon=True)
    start_thread.start()

    time.sleep(0.5)
    print("Reading from SO101 Follower SHM (Ctrl+C to stop)...")

    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=15.0)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None:
                arr = data.get("qpos")
                if arr is not None:
                    print(f"  qpos: {arr}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        start_thread.join(timeout=2.0)
        print("Done.")
