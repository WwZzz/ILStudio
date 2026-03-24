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
    - unsqueeze (side grip): Enables pose delta transmission.
      Hold to move the robot arm. When released, pose deltas are zeroed.
      In code: is_unsqueeze_pressed, _left_unsqueeze_active, _right_unsqueeze_active
      
    - GRIPPER CONTROL:
      * gripper_button_control=True: A/X button opens, TRIGGER closes
      * gripper_button_control=False: TRIGGER controls gripper (original mode)
    
    Action format:
        Single arm: [dx, dy, dz, droll, dpitch, dyaw, gripper] (7D)
        Dual arm: [left_dx, left_dy, left_dz, left_droll, left_dpitch, left_dyaw, left_gripper,
                   right_dx, right_dy, right_dz, right_droll, right_dpitch, right_dyaw, right_gripper] (14D)
    """
    
    # Supported teleop modes
    TELEOP_MODES = ("delta_ee", "rel_ee")
    
    def __init__(
        self,
        name: str = "quest3_teleop",
        max_size_mb: int = 1,
        fps: float = 50.0,
        mode: str = 'delta_ee',
        arm_mode: Literal["left", "right", "dual"] = "dual",
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        connect_timeout: float = 60.0,
        calibration_transform: Optional[list] = None,
        gripper_default_closed: bool = True,
        gripper_delta_mode: bool = False,
        gripper_delta_value: float = 1.0,
        gripper_button_control: bool = False,  # Use A/B (right) or X/Y (left) buttons instead of trigger
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize Quest3 VR teleoperator.
        
        Args:
            name: Name for shared memory
            max_size_mb: Max shared memory size in MB
            fps: Control frequency in Hz
            mode: Teleop mode:
                  - "delta_ee": Frame-to-frame delta mode. Each action is the pose change 
                                between consecutive frames.
                  - "rel_ee": Relative to anchor mode. When unsqueeze is activated, the VR 
                              pose at that moment becomes the anchor. All subsequent actions 
                              are relative poses from the anchor until re-freeze.
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
            gripper_button_control: If True, use A/B buttons (right hand) or X/Y buttons (left hand)
                                   to control gripper instead of trigger.
                                   Right hand: A=open delta, B=close delta
                                   Left hand: X=open delta, Y=close delta
                                   Requires gripper_delta_mode=True
            debug: Enable debug output
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps)
        
        # Validate and store teleop mode
        if mode not in self.TELEOP_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.TELEOP_MODES}")
        self.mode = mode
        
        self.arm_mode = arm_mode
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.connect_timeout = connect_timeout
        self.gripper_default_closed = gripper_default_closed
        self.gripper_delta_mode = gripper_delta_mode
        self.gripper_delta_value = gripper_delta_value
        self.gripper_button_control = gripper_button_control
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
        
        # State tracking for delta calculation (delta_ee mode: prev frame, rel_ee mode: anchor)
        self._left_prev_position = None
        self._left_prev_quaternion = None
        self._left_unsqueeze_active = False  # VR side grip state - enables pose transmission
        self._left_last_goal_id = None  # Track if goal changed
        
        self._right_prev_position = None
        self._right_prev_quaternion = None
        self._right_unsqueeze_active = False  # VR side grip state - enables pose transmission
        self._right_last_goal_id = None  # Track if goal changed
        
        # B button go-home: edge detection (only trigger once per press)
        self._right_b_was_pressed = False
        self._left_b_was_pressed = False
        
        # Anchor positions for rel_ee mode (set when unsqueeze is first activated)
        self._left_anchor_position = None
        self._left_anchor_quaternion = None
        self._right_anchor_position = None
        self._right_anchor_quaternion = None
        
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
        unsqueeze_was_active: bool,
        is_right_hand: bool = True,
        anchor_position: Optional[np.ndarray] = None,
        anchor_quaternion: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute pose delta for one arm.
        
        Args:
            prev_position: Previous frame position (for delta_ee mode reference tracking)
            prev_quaternion: Previous frame quaternion (for delta_ee mode reference tracking)
            unsqueeze_was_active: Whether VR side grip was active last frame
            is_right_hand: True for right hand (A button), False for left hand (X button)
            anchor_position: Anchor position for rel_ee mode (set when unsqueeze first activated)
            anchor_quaternion: Anchor quaternion for rel_ee mode
        
        Returns:
            (delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_pos, new_anchor_quat)
            
            - delta_pose/rel_pose: [dx, dy, dz, droll, dpitch, dyaw] (6D)
              - delta_ee mode: Frame-to-frame change, only non-zero when unsqueeze active
              - rel_ee mode: Relative to anchor pose, only non-zero when unsqueeze active
            - new_unsqueeze_active: VR side grip state (for pose transmission)
            - gripper_value: Robot GRIPPER control value
            - new_anchor_pos/quat: Updated anchor for rel_ee mode (only set on unsqueeze activation)
        """
        delta_pose = np.zeros(6)
        # Default gripper value: no change in button control mode
        gripper_value = 0.0
        new_position = prev_position
        new_quaternion = prev_quaternion
        new_unsqueeze_active = unsqueeze_was_active
        new_anchor_position = anchor_position
        new_anchor_quaternion = anchor_quaternion
        
        if goal is None:
            return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion
        
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
        # Squeeze (side grip) - unfreezes pose delta transmission
        # When pressed, position/rotation deltas are computed and sent
        # This is NOT the robot gripper! This is the VR controller's side button.
        # ============================================================
        is_unsqueeze_pressed = False
        if goal.metadata:
            # Check grip_active field (VR side grip state from XLeVR)
            is_unsqueeze_pressed = goal.metadata.get("grip_active", False)
            
            # Also check buttons dict for unsqueeze
            buttons = goal.metadata.get("buttons", {})
            
            if buttons and buttons.get("unsqueeze", False):
                is_unsqueeze_pressed = True
            
            # ============================================================
            # GRIPPER CONTROL - via trigger or buttons (A/B for right, X/Y for left)
            # ============================================================
            if self.gripper_button_control:
                # Button control mode: 
                # Right hand: A=open, B=close (or trigger=close)
                # Left hand: X=open, Y=close (or trigger=close)
                if is_right_hand:
                    # Right hand: check for A button (open) and B button (close)
                    open_button = (buttons.get("a", False) or 
                                 buttons.get("button_a", False) or
                                 buttons.get("A", False))
                    close_button_btn = (buttons.get("b", False) or 
                                      buttons.get("button_b", False) or
                                      buttons.get("B", False))
                else:
                    # Left hand: check for X button (open) and Y button (close)
                    open_button = (buttons.get("x", False) or 
                                 buttons.get("button_x", False) or
                                 buttons.get("X", False))
                    close_button_btn = (buttons.get("y", False) or 
                                      buttons.get("button_y", False) or
                                      buttons.get("Y", False))
                
                # Use trigger for close (both hands) - trigger can also close
                trigger_active = goal.metadata.get("trigger_active", False)
                if not trigger_active:
                    trigger_value = goal.metadata.get("trigger_value", 0.0)
                    if trigger_value > 0.5:
                        trigger_active = True
                close_button = trigger_active or close_button_btn
                
                # Button control requires delta mode
                if open_button:
                    gripper_value = self.gripper_delta_value   # Open (positive delta)
                elif close_button:
                    gripper_value = -self.gripper_delta_value  # Close (negative delta)
                else:
                    gripper_value = 0.0  # No change when no button pressed
                
            else:
                # Trigger control mode (original behavior)
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
        
        # Handle unsqueeze (side grip) state transitions - enables pose transmission
        if current_position is not None:
            if is_unsqueeze_pressed:
                if not unsqueeze_was_active:
                    # unsqueeze just activated - set current position as reference/anchor
                    # Don't send delta on first frame
                    new_position = current_position.copy()
                    new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    new_unsqueeze_active = True
                    
                    # For rel_ee mode: set anchor position
                    if self.mode == "rel_ee":
                        new_anchor_position = current_position.copy()
                        new_anchor_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion
                else:
                    # unsqueeze held - compute delta/relative pose based on mode
                    if self.mode == "delta_ee" and prev_position is not None:
                        # delta_ee mode: compute delta from previous frame in PREVIOUS FRAME'S LOCAL FRAME
                        # This ensures consistent movement direction relative to hand orientation
                        
                        # Get previous frame rotation matrix (VR world -> prev local)
                        if prev_quaternion is not None:
                            R_prev_world = R.from_quat(prev_quaternion).as_matrix()
                        else:
                            R_prev_world = np.eye(3)
                        
                        # Position delta in VR world frame
                        pos_diff_world = current_position - prev_position
                        
                        # Transform to previous frame's LOCAL frame
                        pos_delta_local = R_prev_world.T @ pos_diff_world
                        pos_delta_local = pos_delta_local * self.position_scale
                        
                        # Apply calibration transform
                        pos_delta_robot = self.calibration_transform @ pos_delta_local
                        delta_pose[:3] = pos_delta_robot
                        
                        # Rotation delta in local frame
                        if current_quaternion is not None and prev_quaternion is not None:
                            try:
                                R_current_world = R.from_quat(current_quaternion).as_matrix()
                                
                                # Relative rotation in prev frame's local:
                                # R_rel_local = R_prev^T @ R_current
                                R_rel_local = R_prev_world.T @ R_current_world
                                
                                # Apply calibration transform
                                T = self.calibration_transform
                                R_rel_robot = T @ R_rel_local @ T.T
                                
                                euler_delta_robot = R.from_matrix(R_rel_robot).as_euler('xyz', degrees=False)
                                delta_pose[3:6] = euler_delta_robot * self.rotation_scale
                            except Exception as e:
                                if self.debug:
                                    print(f"[Quest3Teleop] Rotation delta error: {e}")
                        
                        # Update reference for next frame
                        new_position = current_position.copy()
                        new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                    
                    elif self.mode == "rel_ee" and anchor_position is not None:
                        # rel_ee mode: compute relative pose from anchor in ANCHOR'S LOCAL FRAME
                        # This ensures that hand movement "forward" always maps to robot "forward"
                        # regardless of how the user rotates during freeze
                        
                        # Get anchor rotation matrix (VR world -> anchor local)
                        if anchor_quaternion is not None:
                            R_anchor_world = R.from_quat(anchor_quaternion).as_matrix()
                        else:
                            R_anchor_world = np.eye(3)
                        
                        # Position difference in VR world frame
                        pos_diff_world = current_position - anchor_position
                        
                        # Transform position difference to anchor's LOCAL frame
                        # R_anchor_world.T rotates from world to anchor-local
                        pos_rel_local = R_anchor_world.T @ pos_diff_world
                        pos_rel_local = pos_rel_local * self.position_scale
                        
                        # Apply calibration transform (anchor-local VR -> robot frame)
                        pos_rel_robot = self.calibration_transform @ pos_rel_local
                        delta_pose[:3] = pos_rel_robot
                        
                        # Rotation relative to anchor in anchor's local frame
                        if current_quaternion is not None and anchor_quaternion is not None:
                            try:
                                R_current_world = R.from_quat(current_quaternion).as_matrix()
                                
                                # Relative rotation in anchor's local frame:
                                # R_rel_local = R_anchor^T @ R_current
                                # This represents how the hand has rotated relative to its initial orientation
                                R_rel_local = R_anchor_world.T @ R_current_world
                                
                                # Apply calibration transform to convert to robot frame
                                T = self.calibration_transform
                                R_rel_robot = T @ R_rel_local @ T.T
                                
                                euler_rel_robot = R.from_matrix(R_rel_robot).as_euler('xyz', degrees=False)
                                delta_pose[3:6] = euler_rel_robot * self.rotation_scale
                            except Exception as e:
                                if self.debug:
                                    print(f"[Quest3Teleop] Rotation rel error: {e}")
                        
                        # Update prev position for tracking (but anchor stays fixed)
                        new_position = current_position.copy()
                        new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                        # Anchor remains unchanged until next unsqueeze activation
                    
                    new_unsqueeze_active = True
            else:
                # unsqueeze not pressed - pose frozen, update reference for next activation
                new_position = current_position.copy()
                new_quaternion = current_quaternion.copy() if current_quaternion is not None else None
                new_unsqueeze_active = False
                # In rel_ee mode, clear anchor when unsqueeze is released (will be reset on next activation)
                if self.mode == "rel_ee":
                    new_anchor_position = None
                    new_anchor_quaternion = None
        
        return delta_pose, new_position, new_quaternion, new_unsqueeze_active, gripper_value, new_anchor_position, new_anchor_quaternion

    def _get_unsqueeze_pressed(self, goal) -> bool:
        """Extract current unsqueeze (side grip) state from goal metadata."""
        if goal is None or not goal.metadata:
            return False
        if goal.metadata.get("grip_active", False):
            return True
        buttons = goal.metadata.get("buttons", {})
        return bool(buttons and buttons.get("unsqueeze", False))
    
    def convert_data_to_action(self, data: dict) -> tuple:
        """
        Convert VR data to robot action.
        
        Args:
            data: Dictionary containing VR pose data
            
        Returns:
            (action_dict, should_write): Action dict and whether to write to shm
            action_dict contains:
                - action: The action array
                - unsqueeze_active: Whether VR is currently transmitting (side grip pressed)
                - anchor_just_set: Whether anchor was just set this frame (first frame of unsqueeze)
            should_write is True whenever there's new VR data
            
        Note:
            - unsqueeze (side grip): Enables pose delta transmission
            - Gripper (A/X + trigger): Works independently, always active
        """
        should_write = False
        has_new_data = False
        anchor_just_set = False  # True on first frame of unsqueeze activation
        
        # Track previous unsqueeze states to detect transitions
        left_was_active = self._left_unsqueeze_active
        right_was_active = self._right_unsqueeze_active
        
        if self.arm_mode == "dual":
            # Initialize with default gripper values
            # Button control mode: default is 0 (no change) — only explicit button press produces delta
            # Trigger control mode: default is close/open direction (trigger released = default state)
            if self.gripper_button_control:
                default_gripper = 0.0
            elif self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(14)
            action[6] = default_gripper   # Left gripper default
            action[13] = default_gripper  # Right gripper default
            
            # Left arm - check if new goal OR unsqueeze state change
            left_goal = data.get("left")
            left_goal_id = id(left_goal) if left_goal else None
            left_is_new = (left_goal_id != self._left_last_goal_id) and (left_goal is not None)
            self._left_last_goal_id = left_goal_id
            left_unsqueeze_now = self._get_unsqueeze_pressed(left_goal)
            left_should_update = left_is_new or (left_unsqueeze_now != left_was_active)
            
            if left_should_update:
                (left_delta, self._left_prev_position, self._left_prev_quaternion, 
                 self._left_unsqueeze_active, left_gripper,
                 self._left_anchor_position, self._left_anchor_quaternion) = self._compute_arm_delta(
                    left_goal, 
                    self._left_prev_position, 
                    self._left_prev_quaternion,
                    self._left_unsqueeze_active,
                    is_right_hand=False,  # Left hand: X button
                    anchor_position=self._left_anchor_position,
                    anchor_quaternion=self._left_anchor_quaternion,
                )
                action[0:6] = left_delta
                action[6] = left_gripper
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._left_unsqueeze_active and not left_was_active:
                    anchor_just_set = True
            
            # Right arm - check if new goal OR unsqueeze state change
            right_goal = data.get("right")
            right_goal_id = id(right_goal) if right_goal else None
            right_is_new = (right_goal_id != self._right_last_goal_id) and (right_goal is not None)
            self._right_last_goal_id = right_goal_id
            right_unsqueeze_now = self._get_unsqueeze_pressed(right_goal)
            right_should_update = right_is_new or (right_unsqueeze_now != right_was_active)
            
            if right_should_update:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_unsqueeze_active, right_gripper,
                 self._right_anchor_position, self._right_anchor_quaternion) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_unsqueeze_active,
                    is_right_hand=True,  # Right hand: A button
                    anchor_position=self._right_anchor_position,
                    anchor_quaternion=self._right_anchor_quaternion,
                )
                action[7:13] = right_delta
                action[13] = right_gripper
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._right_unsqueeze_active and not right_was_active:
                    anchor_just_set = True
            
            # Only write when VR unsqueeze is active AND there's new data
            # OR if unsqueeze just turned off (send one last frame to signal robot)
            unsqueeze_active = self._left_unsqueeze_active or self._right_unsqueeze_active
            just_released = (left_was_active and not self._left_unsqueeze_active) or \
                          (right_was_active and not self._right_unsqueeze_active)
                          
            should_write = (has_new_data and unsqueeze_active) or just_released
            
        elif self.arm_mode == "left":
            # Initialize with default gripper value
            if self.gripper_button_control:
                default_gripper = 0.0
            elif self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            
            left_goal = data.get("left")
            left_goal_id = id(left_goal) if left_goal else None
            left_is_new = (left_goal_id != self._left_last_goal_id) and (left_goal is not None)
            self._left_last_goal_id = left_goal_id
            
            left_unsqueeze_now = self._get_unsqueeze_pressed(left_goal)
            left_should_update = left_is_new or (left_unsqueeze_now != left_was_active)
            if left_should_update:
                (left_delta, self._left_prev_position, self._left_prev_quaternion,
                 self._left_unsqueeze_active, left_gripper,
                 self._left_anchor_position, self._left_anchor_quaternion) = self._compute_arm_delta(
                    left_goal,
                    self._left_prev_position,
                    self._left_prev_quaternion,
                    self._left_unsqueeze_active,
                    is_right_hand=False,  # Left hand: X button
                    anchor_position=self._left_anchor_position,
                    anchor_quaternion=self._left_anchor_quaternion,
                )
                action[0:6] = left_delta
                action[6] = left_gripper
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._left_unsqueeze_active and not left_was_active:
                    anchor_just_set = True
            
            unsqueeze_active = self._left_unsqueeze_active
            just_released = left_was_active and not self._left_unsqueeze_active
            should_write = (has_new_data and unsqueeze_active) or just_released
            
        else:  # right
            # Initialize with default gripper value
            if self.gripper_button_control:
                default_gripper = 0.0
            elif self.gripper_delta_mode:
                default_gripper = -self.gripper_delta_value if self.gripper_default_closed else self.gripper_delta_value
            else:
                default_gripper = 0.0 if self.gripper_default_closed else 1.0
            action = np.zeros(7)
            action[6] = default_gripper
            right_goal = data.get("right")
            right_goal_id = id(right_goal) if right_goal else None
            right_is_new = (right_goal_id != self._right_last_goal_id) and (right_goal is not None)
            self._right_last_goal_id = right_goal_id
            
            right_unsqueeze_now = self._get_unsqueeze_pressed(right_goal)
            right_should_update = right_is_new or (right_unsqueeze_now != right_was_active)
            if right_should_update:
                (right_delta, self._right_prev_position, self._right_prev_quaternion,
                 self._right_unsqueeze_active, right_gripper,
                 self._right_anchor_position, self._right_anchor_quaternion) = self._compute_arm_delta(
                    right_goal,
                    self._right_prev_position,
                    self._right_prev_quaternion,
                    self._right_unsqueeze_active,
                    is_right_hand=True,  # Right hand: A button
                    anchor_position=self._right_anchor_position,
                    anchor_quaternion=self._right_anchor_quaternion,
                )
                action[0:6] = right_delta
                action[6] = right_gripper
                has_new_data = True
                
                # Detect anchor just set (unsqueeze just activated)
                if self._right_unsqueeze_active and not right_was_active:
                    anchor_just_set = True
            
            unsqueeze_active = self._right_unsqueeze_active
            just_released = right_was_active and not self._right_unsqueeze_active
            should_write = (has_new_data and unsqueeze_active) or just_released
        
        # ============================================================
        # Gripper is independent: always write when gripper button is active,
        # even when unsqueeze is not pressed (arm is frozen).
        # ============================================================
        final_unsqueeze_for_gripper = unsqueeze_active if 'unsqueeze_active' in dir() else (self._left_unsqueeze_active or self._right_unsqueeze_active)
        if not final_unsqueeze_for_gripper:
            # Check if any gripper button is pressed (A/trigger for open, B/trigger for close)
            gripper_value_in_action = action[6] if self.arm_mode != "dual" else (action[6] or action[13])
            if abs(gripper_value_in_action) > 1e-6:
                should_write = True
        
        # ============================================================
        # B button go-home detection (only when unsqueeze is NOT active)
        # When unsqueeze is active, B button controls gripper (close).
        # When unsqueeze is released, B button triggers go-home command.
        # Uses edge detection: only triggers on rising edge (press, not hold).
        # ============================================================
        go_home = False
        final_unsqueeze = unsqueeze_active if 'unsqueeze_active' in dir() else (self._left_unsqueeze_active or self._right_unsqueeze_active)
        
        if not final_unsqueeze:
            # Check B button on right hand (or Y on left hand)
            for hand_key in (["right"] if self.arm_mode == "right" else 
                            ["left"] if self.arm_mode == "left" else 
                            ["right", "left"]):
                goal = data.get(hand_key)
                if goal is not None and goal.metadata:
                    buttons = goal.metadata.get("buttons", {})
                    if hand_key == "right":
                        b_pressed = (buttons.get("b", False) or 
                                    buttons.get("button_b", False) or
                                    buttons.get("B", False))
                        # Rising edge detection
                        if b_pressed and not self._right_b_was_pressed:
                            go_home = True
                        self._right_b_was_pressed = b_pressed
                    else:
                        b_pressed = (buttons.get("y", False) or 
                                    buttons.get("button_y", False) or
                                    buttons.get("Y", False))
                        if b_pressed and not self._left_b_was_pressed:
                            go_home = True
                        self._left_b_was_pressed = b_pressed
        else:
            # Reset edge detection when unsqueeze is active (B is used for gripper)
            self._right_b_was_pressed = False
            self._left_b_was_pressed = False
        
        if go_home:
            should_write = True
        
        # Return action dict with state information for robot side anchor management
        action_dict = {
            "action": action,
            "unsqueeze_active": final_unsqueeze,
            "anchor_just_set": anchor_just_set,
            "go_home": go_home,
        }
        
        return action_dict, should_write
    
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
                    action_dict, should_write = self.convert_data_to_action(data)
                    # Only write to shm when unsqueeze button is pressed AND there's new VR data
                    if should_write:
                        # Write action dict with state info (action, unsqueeze_active, anchor_just_set)
                        self.write_data_to_shm(action_dict)
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
    parser.add_argument("--test-buttons", action="store_true",
                        help="Test button data capture (direct VR access)")
    args = parser.parse_args()
    
    # Button test mode: directly access VR data without starting device process
    if args.test_buttons:
        print("=== Button Test Mode ===")
        print("Press buttons on VR controller to see their states.")
        print("Available buttons: trigger, a, b, x, y, unsqueeze (side grip)\n")
        
        # Create a test instance to access VR data directly
        test_teleop = Quest3Teleop(
            name="quest3_test",
            arm_mode=args.mode,
            debug=True,
            gripper_button_control=True,
            gripper_delta_mode=True,
        )
        
        if not test_teleop.connect():
            print("Failed to connect to VR. Exiting.")
            sys.exit(1)
        
        try:
            last_print_time = 0
            print_interval = 0.1  # Print every 100ms
            
            while True:
                vr_data = test_teleop.get_data()
                current_time = time.time()
                
                # Collect data from both hands
                right_data = None
                left_data = None
                
                if vr_data is not None:
                    # Check right controller buttons
                    if vr_data.get("has_right") and vr_data.get("right"):
                        right_goal = vr_data["right"]
                        if right_goal and right_goal.metadata:
                            buttons = right_goal.metadata.get("buttons", {})
                            right_data = {
                                "buttons": buttons,
                                "trigger_active": right_goal.metadata.get("trigger_active", False),
                                "trigger_value": right_goal.metadata.get("trigger_value", 0.0),
                                "grip_active": right_goal.metadata.get("grip_active", False),
                            }
                    
                    # Check left controller buttons
                    if vr_data.get("has_left") and vr_data.get("left"):
                        left_goal = vr_data["left"]
                        if left_goal and left_goal.metadata:
                            buttons = left_goal.metadata.get("buttons", {})
                            left_data = {
                                "buttons": buttons,
                                "trigger_active": left_goal.metadata.get("trigger_active", False),
                                "trigger_value": left_goal.metadata.get("trigger_value", 0.0),
                                "grip_active": left_goal.metadata.get("grip_active", False),
                            }
                
                # Print both hands together every 100ms
                if current_time - last_print_time >= print_interval:
                    print("\n" + "="*70)
                    print("DUAL HAND BUTTON STATES")
                    print("="*70)
                    
                    # Right hand
                    if right_data:
                        print("\nRIGHT CONTROLLER:")
                        print(f"  trigger_active: {right_data['trigger_active']}")
                        print(f"  trigger_value: {right_data['trigger_value']:.3f}")
                        print(f"  grip_active (unsqueeze): {right_data['grip_active']}")
                        print(f"  buttons dict: {right_data['buttons']}")
                        if right_data['buttons']:
                            print("  Individual buttons:")
                            for btn_name, btn_state in right_data['buttons'].items():
                                status = "✓" if btn_state else "✗"
                                print(f"    {status} {btn_name}: {btn_state}")
                        else:
                            print("  WARNING: buttons dict is empty!")
                    else:
                        print("\nRIGHT CONTROLLER: Not available")
                    
                    # Left hand
                    if left_data:
                        print("\nLEFT CONTROLLER:")
                        print(f"  trigger_active: {left_data['trigger_active']}")
                        print(f"  trigger_value: {left_data['trigger_value']:.3f}")
                        print(f"  grip_active (unsqueeze): {left_data['grip_active']}")
                        print(f"  buttons dict: {left_data['buttons']}")
                        if left_data['buttons']:
                            print("  Individual buttons:")
                            for btn_name, btn_state in left_data['buttons'].items():
                                status = "✓" if btn_state else "✗"
                                print(f"    {status} {btn_name}: {btn_state}")
                        else:
                            print("  WARNING: buttons dict is empty!")
                    else:
                        print("\nLEFT CONTROLLER: Not available")
                    
                    print("="*70)
                    last_print_time = current_time
                
                time.sleep(0.01)  # Small sleep to avoid CPU spinning
        except KeyboardInterrupt:
            print("\nStopping button test...")
        finally:
            test_teleop.close()
        sys.exit(0)
    
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
    print("Press and hold unsqueeze button on VR controller to send pose deltas.\n")
    
    try:
        shm = SharedMemoryChannel(shm_name, is_writer=False, timeout=args.timeout + 10)
        while True:
            data = shm.read(blocking=True, timeout=1.0)
            if data is not None and "action" in data:
                arr = data["action"]
                if len(arr) >= 14:
                    # Dual arm mode
                    print(
                        f"L: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] rot:[{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] g:{arr[6]:.3f} | "
                        f"R: [{arr[7]:+.3f},{arr[8]:+.3f},{arr[9]:+.3f}] rot:[{arr[10]:+.2f},{arr[11]:+.2f},{arr[12]:+.2f}] g:{arr[13]:.3f}",
                        end="\r",
                        flush=True,
                    )
                elif len(arr) >= 7:
                    # Single arm mode
                    print(
                        f"pos: [{arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f}] "
                        f"rot: [{arr[3]:+.2f},{arr[4]:+.2f},{arr[5]:+.2f}] "
                        f"grip: {arr[6]:.3f}",
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
