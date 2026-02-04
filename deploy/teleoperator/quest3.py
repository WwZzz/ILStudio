#!/usr/bin/env python3
"""
Quest3 VR Teleoperator - VR teleoperation controller for ILStudio
Uses Quest3 VR headset for robot teleoperation via XLeVR
"""

import os
import sys
import asyncio
import threading
import time
import numpy as np
from typing import Optional, Literal, Dict, Any
from scipy.spatial.transform import Rotation as R

from deploy.teleoperator.base import BaseTeleopDevice
from deploy.utils import RateLimiter

# Set the absolute path to the XLeVR folder
XLEVR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'XLeVR'))


def setup_xlevr_environment():
    """Setup XLeVR environment"""
    if XLEVR_PATH not in sys.path:
        sys.path.insert(0, XLEVR_PATH)
    os.chdir(XLEVR_PATH)
    os.environ['PYTHONPATH'] = f"{XLEVR_PATH}:{os.environ.get('PYTHONPATH', '')}"


class Quest3Teleop(BaseTeleopDevice):
    """
    Quest3 VR Teleoperator for ILStudio.
    
    Captures VR controller poses and converts them to end-effector pose deltas.
    
    VR Controller Button Mapping:
    =============================
    - SQUEEZE (side grip button): Enables pose delta transmission. 
      Hold this button to move the robot arm. When released, no pose data is sent.
      In code: is_squeeze_pressed, squeeze_active, _left_squeeze_active, _right_squeeze_active
      
    - TRIGGER (front trigger): Controls the robot's GRIPPER (end-effector jaw).
      Press to open/close the gripper depending on gripper_default_closed setting.
      In code: trigger_active, trigger_value
      
    Note: "Squeeze" is the VR button name, "Gripper" refers to the robot's jaw.
    
    Action format:
        Single arm: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7D)
        Dual arm: [left_dx, left_dy, left_dz, left_droll, left_dpitch, left_dyaw, left_gripper,
                   right_dx, right_dy, right_dz, right_droll, right_dpitch, right_dyaw, right_gripper] (14D)
    """
    
    def __init__(
        self,
        name: str = "quest3_teleop",
        max_size_mb: int = 1,
        fps: float = 50.0,
        arm_mode: Literal["left", "right", "dual"] = "dual",
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        connect_timeout: float = 60.0,
        calibration_transform: Optional[list] = None,
        gripper_default_closed: bool = True,
        gripper_delta_mode: bool = False,
        gripper_delta_value: float = 1.0,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize Quest3 VR teleoperator.
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency in Hz
            arm_mode: "left", "right", or "dual" for which arm(s) to control
            position_scale: Scale factor for position delta
            rotation_scale: Scale factor for rotation delta
            connect_timeout: Timeout in seconds for VR connection
            calibration_transform: 3x3 matrix to transform VR coordinates to robot coordinates
                                   Applied to both position (as linear transform) and rotation
                                   (as similarity transform: R_robot = T @ R_vr @ T^T)
                                   This matrix handles both axis remapping AND sign corrections.
                                   If None, no transformation is applied (identity matrix)
            gripper_default_closed: If True, gripper is closed by default, trigger opens it
                                    If False, gripper is open by default, trigger closes it
            gripper_delta_mode: If True, gripper outputs delta values for incremental control
                               (positive=open more, negative=close more)
                               If False, gripper outputs absolute state (0=closed, 1=open)
            gripper_delta_value: The delta value to use when gripper_delta_mode is True
                                Default: 1.0 (adjust based on robot's gripper speed)
            debug: Enable debug output
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)
        
        self.arm_mode = arm_mode
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.connect_timeout = connect_timeout
        self.gripper_default_closed = gripper_default_closed
        self.gripper_delta_mode = gripper_delta_mode
        self.gripper_delta_value = gripper_delta_value
        self.debug = debug
        
        # Calibration transform matrix (3x3)
        # Used for both position and rotation coordinate transformation
        # Applied AFTER left-handed to right-handed conversion (if enabled)
        # This matrix should include both axis remapping AND any sign corrections
        if calibration_transform is not None:
            self.calibration_transform = np.array(calibration_transform, dtype=np.float64)
            if self.calibration_transform.shape != (3, 3):
                raise ValueError(f"calibration_transform must be 3x3 matrix, got {self.calibration_transform.shape}")
        else:
            self.calibration_transform = np.eye(3)  # Identity matrix (no transform)
        
        # VR Monitor instance
        self.vr_monitor = None
        self._vr_thread = None
        self._async_loop = None
        
        # State tracking for delta calculation
        self._left_prev_position = None
        self._left_prev_quaternion = None
        self._left_squeeze_active = False  # VR SQUEEZE button state (not robot gripper!)
        self._left_last_goal_id = None  # Track if goal changed
        
        self._right_prev_position = None
        self._right_prev_quaternion = None
        self._right_squeeze_active = False  # VR SQUEEZE button state (not robot gripper!)
        self._right_last_goal_id = None  # Track if goal changed
        
        # Robot GRIPPER states (0.0 = open, 1.0 = closed) - controlled by VR TRIGGER
        self._left_gripper = 0.0
        self._right_gripper = 0.0
        
        # Action dimension
        if arm_mode == "dual":
            self.action_dim = 14  # 7 per arm
        else:
            self.action_dim = 7
    
    def _run_vr_monitor_async(self):
        """Run VR monitor in a separate thread with its own event loop."""
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        
        try:
            self._async_loop.run_until_complete(self.vr_monitor.start_monitoring())
        except Exception as e:
            if self.debug:
                print(f"[Quest3Teleop] VR monitor stopped: {e}")
        finally:
            self._async_loop.close()
    
    def connect(self) -> bool:
        """
        Connect to VR headset and wait for pose data.
        
        Returns:
            True if connected successfully, False otherwise
        """
        print("[Quest3Teleop] Initializing VR connection...")
        
        # Setup XLeVR environment
        setup_xlevr_environment()
        
        # Import VRMonitor from the local module
        from vr_monitor import VRMonitor
        
        # Create VR monitor
        self.vr_monitor = VRMonitor()
        
        # Start VR monitor in background thread
        self._vr_thread = threading.Thread(target=self._run_vr_monitor_async, daemon=True)
        self._vr_thread.start()
        
        # Wait for VR connection (check for headset data)
        print("[Quest3Teleop] Waiting for VR headset connection...")
        start_time = time.time()
        
        while time.time() - start_time < self.connect_timeout:
            # Check if headset data is available
            goals = self.vr_monitor.get_latest_goal_nowait()
            if goals and goals.get("has_headset", False):
                print("[Quest3Teleop] VR headset connected successfully!")
                return True
            
            time.sleep(0.1)
        
        print(f"[Quest3Teleop] VR connection timeout after {self.connect_timeout}s")
        return False
    
    def get_data(self) -> Optional[dict]:
        """
        Get VR controller data.
        
        Returns:
            Dictionary containing VR pose data, or None if not available
        """
        if self.vr_monitor is None:
            return None
        
        goals = self.vr_monitor.get_latest_goal_nowait()
        if goals is None:
            return None
        
        return {
            "left": goals.get("left"),
            "right": goals.get("right"),
            "headset": goals.get("headset"),
            "has_left": goals.get("has_left", False),
            "has_right": goals.get("has_right", False),
            "has_headset": goals.get("has_headset", False),
        }
    
    def _compute_arm_delta(
        self,
        goal,
        prev_position: Optional[np.ndarray],
        prev_quaternion: Optional[np.ndarray],
        squeeze_was_active: bool
    ) -> tuple:
        """
        Compute pose delta for one arm.
        
        Args:
            squeeze_was_active: Whether VR SQUEEZE button (side grip) was active last frame
        
        Returns:
            (delta_pose, new_position, new_quaternion, new_squeeze_active, gripper_value)
            - delta_pose: [dx, dy, dz, droll, dpitch, dyaw] (6D)
            - new_squeeze_active: VR SQUEEZE button state (for pose transmission)
            - gripper_value: Robot GRIPPER (jaw) control value
        """
        delta_pose = np.zeros(6)
        # Set default robot gripper value based on mode and settings
        # NOTE: gripper_value controls the ROBOT's jaw, not the VR squeeze button!
        if self.gripper_delta_mode:
            # Delta mode: negative = close, positive = open
            gripper_value = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
        else:
            # Absolute mode: 0.0 = closed, 1.0 = open
            gripper_value = 0.0 if self.gripper_default_closed else 1.0
        new_position = prev_position
        new_quaternion = prev_quaternion
        new_squeeze_active = squeeze_was_active
        
        if goal is None:
            return delta_pose, new_position, new_quaternion, new_squeeze_active, gripper_value
        
        # Get current position (use vr_position for raw VR coordinates)
        current_position = None
        if goal.metadata and "vr_position" in goal.metadata:
            # Use raw VR position for delta calculation
            current_position = np.array(goal.metadata["vr_position"])
        elif goal.target_position is not None:
            current_position = np.array(goal.target_position)
        
        # Get current quaternion from metadata
        current_quaternion = None
        if goal.metadata and "quaternion" in goal.metadata:
            quat = goal.metadata["quaternion"]
            if quat and isinstance(quat, dict) and all(k in quat for k in ['x', 'y', 'z', 'w']):
                # Quaternion format: [x, y, z, w]
                current_quaternion = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
        
        # ============================================================
        # VR SQUEEZE BUTTON (side grip) - enables pose delta transmission
        # This is NOT the robot gripper! This is the VR controller's side button.
        # ============================================================
        is_squeeze_pressed = False
        if goal.metadata:
            # Check grip_active field (VR squeeze button state from XLeVR)
            is_squeeze_pressed = goal.metadata.get("grip_active", False)
            
            # Also check buttons dict for squeeze
            buttons = goal.metadata.get("buttons", {})
            if buttons and buttons.get("squeeze", False):
                is_squeeze_pressed = True
            
            # ============================================================
            # VR TRIGGER BUTTON (front trigger) - controls ROBOT GRIPPER
            # trigger_active -> controls gripper_value (robot's jaw)
            # ============================================================
            trigger_active = goal.metadata.get("trigger_active", False)
            
            # Also check buttons dict for trigger
            if not trigger_active and buttons:
                trigger_active = buttons.get("trigger", False)
            
            # Also check trigger_value (some VR systems report as float 0-1)
            trigger_value = goal.metadata.get("trigger_value", 0.0)
            if not trigger_active and trigger_value > 0.5:
                trigger_active = True
            
            if self.debug:
                print(f"[Quest3Teleop] trigger_active={trigger_active}, trigger_value={trigger_value}")
            
            if self.gripper_delta_mode:
                # Delta mode: output incremental values for robots using delta control
                # Positive = open more, Negative = close more
                if self.gripper_default_closed:
                    # Default closing, trigger opens
                    if trigger_active:
                        gripper_value = self.gripper_delta_value   # Open (positive delta)
                    else:
                        gripper_value = -self.gripper_delta_value  # Close (negative delta)
                else:
                    # Default opening, trigger closes
                    if trigger_active:
                        gripper_value = -self.gripper_delta_value  # Close (negative delta)
                    else:
                        gripper_value = self.gripper_delta_value   # Open (positive delta)
            else:
                # Absolute mode: output state values (0=closed, 1=open)
                if self.gripper_default_closed:
                    # Default closed, trigger opens
                    if trigger_active:
                        gripper_value = 1.0  # Open when trigger pressed
                    else:
                        gripper_value = 0.0  # Closed when trigger released
                else:
                    # Default open, trigger closes
                    if trigger_active:
                        gripper_value = 0.0  # Closed when trigger pressed
                    else:
                        gripper_value = 1.0  # Open when trigger released
        
        # Handle VR SQUEEZE button state transitions (controls pose transmission)
        if current_position is not None:
            if is_squeeze_pressed:
                if not squeeze_was_active:
                    # Squeeze just activated - set current position as reference
                    # Don't send delta on first frame of squeeze
                    new_position = current_position.copy()
                    new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    new_squeeze_active = True
                    if self.debug:
                        print(f"[Quest3Teleop] Squeeze activated - setting reference position")
                    return delta_pose, new_position, new_quaternion, new_squeeze_active, gripper_value
                else:
                    # Squeeze held - compute delta from reference
                    if prev_position is not None:
                        """
                        # 1. 位置增量转换
                        pos_delta_vr = current_position - prev_position
                        pos_delta_robot = T @ pos_delta_vr * self.position_scale

                        # 2. 旋转增量转换 (更稳健的方法)
                        r_curr = R.from_quat(current_quaternion)
                        r_prev = R.from_quat(prev_quaternion)

                        # 计算在机器人参考系下的相对旋转
                        # 这种方法避免了手动处理左/右手系的复杂性
                        # 它是把“VR里的姿态变化”映射到“机器人空间”
                        rot_delta_mat = T @ (r_curr * r_prev.inv()).as_matrix() @ T.T
                        euler_delta = R.from_matrix(rot_delta_mat).as_euler('xyz')
                        """
                        # Position delta in VR frame
                        pos_delta_vr = (current_position - prev_position) * self.position_scale
                        
                        # Apply coordinate transform to position (linear transform)
                        pos_delta_robot = self.calibration_transform @ pos_delta_vr
                        delta_pose[:3] = pos_delta_robot
                        
                        # Rotation delta - use rotation matrix as intermediate representation
                        # Euler angles cannot be directly transformed with matrices
                        if current_quaternion is not None and prev_quaternion is not None:
                            try:
                                T = self.calibration_transform
                                # 推荐做法：鲁棒性更高，自动处理所有坐标系不对齐问题
                                curr_abs_robot = T @ R.from_quat(current_quaternion).as_matrix() @ T.T
                                prev_abs_robot = T @ R.from_quat(prev_quaternion).as_matrix() @ T.T

                                # 在机器人坐标系这个“静止参考系”下计算真实的旋转差
                                rel_rot_robot = curr_abs_robot @ prev_abs_robot.T
                                euler_delta_robot = R.from_matrix(rel_rot_robot).as_euler('xyz', degrees=False)

                                # Compute relative rotation in VR frame
                                # prev_rot = R.from_quat(prev_quaternion)
                                # curr_rot = R.from_quat(current_quaternion)
                                # relative_rot_vr = curr_rot * prev_rot.inv()
                                
                                # # Convert to rotation matrix
                                # rot_matrix_vr = relative_rot_vr.as_matrix()
                                
                                # # Apply coordinate transform (similarity transform)
                                # # R_robot = T @ R_vr @ T^T (for orthogonal T, T^(-1) = T^T)
                                # T = self.calibration_transform
                                # rot_matrix_robot = T @ rot_matrix_vr @ T.T
                                
                                # # Convert back to euler angles in robot frame
                                # rot_robot = R.from_matrix(rot_matrix_robot)
                                # euler_delta_robot = rot_robot.as_euler('xyz', degrees=False)
                                
                                delta_pose[3:6] = euler_delta_robot * self.rotation_scale
                            except Exception as e:
                                if self.debug:
                                    print(f"[Quest3Teleop] Rotation delta error: {e}")
                    
                    # Update reference for next frame
                    new_position = current_position.copy()
                    new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    new_squeeze_active = True
            else:
                # Squeeze not pressed - reset reference, delta stays zero
                new_position = current_position.copy()
                new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                new_squeeze_active = False
        
        return delta_pose, new_position, new_quaternion, new_squeeze_active, gripper_value
    
    def convert_data_to_action(self, data: dict) -> tuple:
        """
        Convert VR data to robot action.
        
        Args:
            data: Dictionary containing VR pose data
            
        Returns:
            (action, should_write): Action array and whether to write to shm
            should_write is True only when VR SQUEEZE button is pressed AND there's new VR data
        """
        should_write = False
        has_new_data = False
        
        if self.arm_mode == "dual":
            # Initialize with default gripper values
            if self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(14)
            action[6] = default_gripper   # Left gripper default
            action[13] = default_gripper  # Right gripper default
            
            # Left arm - check if new goal
            left_goal = data.get("left")
            left_goal_id = id(left_goal) if left_goal else None
            left_is_new = (left_goal_id != self._left_last_goal_id) and (left_goal is not None)
            self._left_last_goal_id = left_goal_id
            
            if left_is_new:
                (left_delta, self._left_prev_position, self._left_prev_quaternion, 
                 self._left_squeeze_active, left_gripper) = self._compute_arm_delta(
                    left_goal, 
                    self._left_prev_position, 
                    self._left_prev_quaternion,
                    self._left_squeeze_active
                )
                action[0:6] = left_delta
                action[6] = left_gripper
                has_new_data = True
            
            # Right arm - check if new goal
            right_goal = data.get("right")
            right_goal_id = id(right_goal) if right_goal else None
            right_is_new = (right_goal_id != self._right_last_goal_id) and (right_goal is not None)
            self._right_last_goal_id = right_goal_id
            
            if right_is_new:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_squeeze_active, right_gripper) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_squeeze_active
                )
                action[7:13] = right_delta
                action[13] = right_gripper
                has_new_data = True
            
            # Only write when VR SQUEEZE is active AND there's new data
            should_write = has_new_data and (self._left_squeeze_active or self._right_squeeze_active)
            
        elif self.arm_mode == "left":
            # Initialize with default gripper value
            if self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            
            left_goal = data.get("left")
            left_goal_id = id(left_goal) if left_goal else None
            left_is_new = (left_goal_id != self._left_last_goal_id) and (left_goal is not None)
            self._left_last_goal_id = left_goal_id
            
            if left_is_new:
                (left_delta, self._left_prev_position, self._left_prev_quaternion,
                 self._left_squeeze_active, left_gripper) = self._compute_arm_delta(
                    left_goal,
                    self._left_prev_position,
                    self._left_prev_quaternion,
                    self._left_squeeze_active
                )
                action[0:6] = left_delta
                action[6] = left_gripper
                has_new_data = True
            
            should_write = has_new_data and self._left_squeeze_active
            
        else:  # right
            # Initialize with default gripper value
            if self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            right_goal = data.get("right")
            right_goal_id = id(right_goal) if right_goal else None
            right_is_new = (right_goal_id != self._right_last_goal_id) and (right_goal is not None)
            self._right_last_goal_id = right_goal_id
            
            if right_is_new:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_squeeze_active, right_gripper) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_squeeze_active
                )
                action[0:6] = right_delta
                action[6] = right_gripper
                has_new_data = True
            
            should_write = has_new_data and self._right_squeeze_active
        
        if self.debug and should_write:
            # Print action only when writing
            if self.arm_mode == "dual":
                print(f"[Quest3Teleop] L: pos=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}] "
                      f"rot=[{action[3]:.3f},{action[4]:.3f},{action[5]:.3f}] grip={action[6]:.1f} | "
                      f"R: pos=[{action[7]:.3f},{action[8]:.3f},{action[9]:.3f}] "
                      f"rot=[{action[10]:.3f},{action[11]:.3f},{action[12]:.3f}] grip={action[13]:.1f}")
            else:
                print(f"[Quest3Teleop] pos=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}] "
                      f"rot=[{action[3]:.3f},{action[4]:.3f},{action[5]:.3f}] grip={action[6]:.1f}")
        
        return action, should_write
    
    def start(self):
        """Start the Quest3 teleoperator."""
        import signal
        
        print(f"[Quest3Teleop] Starting, name={self.name}, mode={self.arm_mode}")
        
        # Connect to VR first
        if not self.connect():
            raise RuntimeError("[Quest3Teleop] Failed to connect to VR headset")
        
        # Create shared memory for output
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Setup signal handler
        def cleanup_handler(signum, frame):
            self.close()
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        
        # Main loop
        self.is_running = True
        rate_limiter = RateLimiter()
        self._write_count = 0
        
        try:
            while self.is_running:
                data = self.get_data()
                
                if data is not None:
                    action, should_write = self.convert_data_to_action(data)
                    # Only write to shm when squeeze button is pressed AND there's new VR data
                    if should_write:
                        self.write_data_to_shm({"action": action})
                        self._write_count += 1
                        
                rate_limiter.sleep(self.fps)
        finally:
            self.close()
    
    def close(self):
        """Close the teleoperator."""
        self.is_running = False
        
        # Stop VR monitor
        if self.vr_monitor is not None:
            self.vr_monitor.is_running = False
        
        # Wait for VR thread to finish
        if self._vr_thread is not None and self._vr_thread.is_alive():
            self._vr_thread.join(timeout=2.0)
        
        super().close()
        print("[Quest3Teleop] Closed")


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    from deploy.base import start_device
    from deploy.shm_utils import SharedMemoryChannel
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quest3 VR Teleoperator")
    parser.add_argument("--mode", "-m", type=str, default="right", 
                        choices=["left", "right", "dual"],
                        help="Arm mode: left, right, or dual (default: right)")
    parser.add_argument("--timeout", "-t", type=float, default=60.0,
                        help="VR connection timeout in seconds (default: 60)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug output")
    args = parser.parse_args()
    
    print(f"Starting Quest3 Teleop in {args.mode.upper()} mode...")
    
    # Device configuration
    device_config = {
        "type": "deploy.teleoperator.quest3.Quest3Teleop",
        "name": "quest3_teleop",
        "args": {
            "name": "quest3_teleop",
            "fps": 50.0,
            "arm_mode": args.mode,
            "position_scale": 1.0,
            "rotation_scale": 1.0,
            "connect_timeout": args.timeout,
            "debug": args.debug,
        }
    }
    
    shm_name = device_config["args"]["name"]
    proc = mp.Process(target=start_device, args=(device_config,))
    proc.start()
    
    time.sleep(2.0)
    print("Reading from SHM (Ctrl+C to stop)...")
    print("Press and hold SQUEEZE button on VR controller to send pose deltas.\n")
    
    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=args.timeout + 10)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                if len(arr) >= 14:
                    # Dual arm mode
                    print(
                        f"L: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] rot:[{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] g:{arr[6]:.0f} | "
                        f"R: [{arr[7]:+.3f},{arr[8]:+.3f},{arr[9]:+.3f}] rot:[{arr[10]:+.2f},{arr[11]:+.2f},{arr[12]:+.2f}] g:{arr[13]:.0f}",
                        end="\r",
                        flush=True,
                    )
                elif len(arr) >= 7:
                    # Single arm mode
                    print(
                        f"pos: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] "
                        f"rot: [{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] "
                        f"grip: {arr[6]:.0f}",
                        end="\r",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
        print("Done.")
