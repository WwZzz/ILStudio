"""
Alicia-D Robot for ILStudio

Integrates Alicia-D robot arm with ILStudio framework using:
- Official alicia_d_sdk for hardware communication (via pip install)
- Piper-style global IK (Pinocchio + scipy / optional casadi+ipopt)
- SharedMemory for observation/action communication

Control modes:
1. qpos: Direct joint position control (6D arm joints + 1D gripper)
2. delta_ee: 7D delta end-effector control with IK (dx, dy, dz, drx, dry, drz, dgripper)
3. rel_ee: 7D relative end-effector control with IK (relative to anchor pose)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from deploy.robot.base import BaseRobot
from benchmark.base import MetaObs
from loguru import logger


def _transform_matrix_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform matrix from xyz + rpy."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    T = np.eye(4)
    T[:3, :3] = rot_z @ rot_y @ rot_x
    T[:3, 3] = xyz
    return T


# ============================================================================
# Alicia-D Robot Class
# ============================================================================

class AliciaD(BaseRobot):
    """
    Alicia-D Robot for ILStudio
    
    Uses alicia_d_sdk.hardware for low-level communication.
    
    Supports three control modes:
    1. "qpos" (default): Direct joint position control (7D: 6 arm + 1 gripper)
    2. "delta_ee": 7D delta EE control with global IK
    3. "rel_ee": 7D relative EE control with IK (relative to anchor pose)
    """
    
    # Control modes
    CONTROL_MODES = ("qpos", "delta_ee", "rel_ee")
    
    def __init__(
        self,
        name: str = "alicia_d",
        max_size_mb: int = 64,
        fps: float = 200.0,
        control_shm_name: Optional[str] = None,
        port: str = "",
        urdf_path: Optional[str] = None,
        control_mode: str = "qpos",
        gripper_type: str = "50mm",
        ik_end_frame: str = "link6",
        ik_ee_offset_xyz: Optional[list] = None,
        ik_ee_offset_rpy: Optional[list] = None,
        ik_ee_offset_matrix: Optional[list] = None,
        ik_eps: float = 2e-3,
        ik_max_iter: int = 100,
        ik_rotation_weight: float = 0.3,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        speed_deg_s: float = 30.0,
        gripper_speed_deg_s: float = 483.4,
        max_joint_delta_deg: float = 10.0,
        debug: bool = False,
        gripper_trace: bool = False,
        **kwargs
    ):
        """Initialize Alicia-D Robot. Extra YAML keys (legacy realtime IK) are ignored."""
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.port = port
        self.debug = debug
        self.gripper_trace = gripper_trace
        self.gripper_type = gripper_type
        
        # Control mode
        if control_mode not in self.CONTROL_MODES:
            raise ValueError(f"Invalid control_mode '{control_mode}'. Must be one of {self.CONTROL_MODES}")
        self.control_mode = control_mode
        
        # Delta EE scaling
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_scale = gripper_scale
        
        # Speed settings
        self.speed_deg_s = speed_deg_s
        self.gripper_speed_deg_s = gripper_speed_deg_s
        
        # Safety settings
        self.max_joint_delta_rad = np.deg2rad(max_joint_delta_deg)
        
        # Find URDF path
        if urdf_path is None:
            # Use local URDF file (bundled with this module)
            local_urdf = Path(__file__).parent / "urdf" / f"Alicia_D_v5_6_gripper_{gripper_type}.urdf"
            if local_urdf.exists():
                urdf_path = str(local_urdf)
            else:
                # Fallback: try synriard if available
                try:
                    from synriard import get_model_path
                    urdf_path = str(get_model_path("Alicia_D", version="v5_6", variant=f"gripper_{gripper_type}", model_format="urdf"))
                except ImportError:
                    raise FileNotFoundError(
                        f"URDF file not found at {local_urdf}. "
                        "Please provide urdf_path argument or install synriard package."
                    )
        self.urdf_path = urdf_path
        
        # Robot driver instance (initialized in connect())
        self._driver = None
        
        if ik_ee_offset_matrix is not None:
            ee_offset_matrix = np.array(ik_ee_offset_matrix, dtype=np.float64)
        else:
            ee_offset_xyz = np.array(ik_ee_offset_xyz or [0.0, 0.0, 0.0], dtype=np.float64)
            ee_offset_rpy = np.array(ik_ee_offset_rpy or [0.0, 0.0, 0.0], dtype=np.float64)
            ee_offset_matrix = _transform_matrix_from_xyz_rpy(ee_offset_xyz, ee_offset_rpy)

        self._ik_solver = None
        self._ik_settings = {
            'end_frame': ik_end_frame,
            'ee_offset_matrix': ee_offset_matrix,
            'eps': ik_eps,
            'max_iter': ik_max_iter,
            'rotation_weight': ik_rotation_weight,
        }
        
        # State tracking
        self._current_joint_angles: Optional[np.ndarray] = None
        self._current_gripper: Optional[float] = None
        self._target_qpos: Optional[np.ndarray] = None
        self._target_gripper: float = 500.0
        self._lock = threading.RLock()  # Use RLock to allow reentrant acquisition (process_action -> refresh_anchor)
        
        # rel_ee mode: anchor pose tracking
        self._rel_ee_anchor_pose: Optional[np.ndarray] = None
        self._rel_ee_anchor_gripper: float = 500.0
        self._rel_ee_vr_unsqueeze_active: bool = False
        self._rel_ee_anchor_initialized: bool = False
        
        # rel_ee mode: anchor offset for dynamic re-anchoring on IK failure.
        # When IK fails, the VR-side relative pose that caused the failure is absorbed
        # into this offset. Subsequent VR relative poses are corrected by subtracting
        # the offset, so the robot effectively re-anchors to its last reachable pose
        # without requiring the user to release and re-squeeze.
        self._rel_ee_pos_offset: np.ndarray = np.zeros(3)   # position offset (robot frame, scaled)
        self._rel_ee_rot_offset: np.ndarray = np.zeros(3)   # euler offset (robot frame, scaled)
        
        # Connection state
        self._connected = False
        self.is_running = False
        
        logger.info(f"[AliciaD] Initialized with control_mode={control_mode}, fps={fps}")
        if control_mode in ("delta_ee", "rel_ee"):
            logger.info(f"[AliciaD] Delta EE scales: pos={position_scale}, rot={rotation_scale}, gripper={gripper_scale}")
            logger.info(
                f"[AliciaD] IK (piper_global): end_frame={ik_end_frame}, "
                f"eps={ik_eps}, max_iter={ik_max_iter}, rotation_weight={ik_rotation_weight}"
            )
    
    @property
    def ik_solver(self):
        """Lazy-initialize Piper-style global IK solver."""
        if self._ik_solver is None:
            from deploy.robot.alicia_d.piper_global_ik import PiperStyleGlobalIKSolver

            s = self._ik_settings
            self._ik_solver = PiperStyleGlobalIKSolver(
                urdf_path=self.urdf_path,
                end_frame=s['end_frame'],
                ee_offset_matrix=s['ee_offset_matrix'],
                position_weight=1.0,
                rotation_weight=max(float(s['rotation_weight']), 1e-6),
                total_cost_scale=20.0,
                regularization_weight=0.01,
                max_iterations=s['max_iter'],
                tol=max(float(s['eps']), 1e-6),
                jump_reset_threshold_deg=np.rad2deg(self.max_joint_delta_rad),
            )
            logger.info(
                f"[AliciaD] IK solver initialized (piper_global): "
                f"eps={s['eps']}, max_iter={s['max_iter']}, end_frame={s['end_frame']}, "
                f"rotation_weight={s['rotation_weight']}"
            )
        return self._ik_solver
    
    def connect(self) -> bool:
        """Connect to robot using low-level hardware SDK."""
        if self._connected:
            return True
        
        try:
            # Import low-level hardware SDK
            import sys
            # Ensure sdk path is in sys.path
            sdk_path = os.path.join(os.path.dirname(__file__))
            if sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)
                
            from alicia_d_sdk.hardware import ServoDriver
            
            # Create driver instance
            logger.info(f"[AliciaD] Connecting via low-level SDK on port '{self.port}'...")
            self._driver = ServoDriver(
                port=self.port,
                debug_mode=self.debug
            )
            
            # Connect
            if not self._driver.connect():
                logger.error("[AliciaD] Failed to connect to robot hardware")
                return False
            self.set_home(speed_deg_s=self.speed_deg_s)
            # Wait for valid state
            logger.info("[AliciaD] Waiting for valid state...")
            if not self._driver.wait_for_valid_state(timeout=2.0):
                 logger.warning("[AliciaD] Initial state read timeout, but continuing...")

            # Get initial state
            # Try to acquire info explicitly to ensure we have data
            self._driver.acquire_info("joint_gripper", wait=True, timeout=1.0)
            state = self._driver.data_parser.get_joint_state()
            
            if state is None or state.angles is None:
                logger.warning("[AliciaD] Initial state read failed")
            else:
                self._current_joint_angles = np.array(state.angles, dtype=np.float64)
                self._current_gripper = state.gripper
                self._target_qpos = self._current_joint_angles.copy()
                self._target_gripper = self._current_gripper
            
            # Initialize IK solver with current joint angles
            if self.control_mode in ("delta_ee", "rel_ee") and self._current_joint_angles is not None:
                self.ik_solver.reset(self._current_joint_angles)
                self.ik_solver.set_reference(self._current_joint_angles)
                logger.info(f"[AliciaD] IK solver initialized with joints (deg): {np.rad2deg(self._current_joint_angles)}")
            
            self._connected = True
            logger.info(f"[AliciaD] Connected to robot successfully")
            if self._current_joint_angles is not None:
                logger.info(f"[AliciaD] Current joints (deg): {np.rad2deg(self._current_joint_angles)}")
            
            return True
            
        except Exception as e:
            logger.error(f"[AliciaD] Connection error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_observation(self) -> Optional[Dict[str, Any]]:
        """
        Get complete observation data (ALWAYS fresh, no caching).
        
        For imitation learning data collection, observations must be
        real-time ground truth, not cached values.
        """
        if not self._connected or self._driver is None:
            return None
        
        try:
            # Actively request joint data from hardware (rate-limited to avoid serial flooding)
            # This ensures we get fresh data, not stale cache from connect() time.
            now = time.time()
            if not hasattr(self, '_last_joint_query_time'):
                self._last_joint_query_time = 0.0
            # Query at most ~50Hz (every 20ms) to avoid flooding serial
            if now - self._last_joint_query_time >= 0.02:
                self._driver.acquire_info("joint_gripper", wait=False)
                self._last_joint_query_time = now
            
            # Get fresh state from SDK (non-blocking read from parser, updated by thread)
            # The ServoDriver has a background thread that updates state
            state = self._driver.data_parser.get_joint_state()
            
            if state is None or state.angles is None:
                return None
            
            joint_angles = np.array(state.angles, dtype=np.float64)
            gripper = state.gripper
            
            # Update internal tracking
            self._current_joint_angles = joint_angles
            self._current_gripper = gripper
            
            # Combine arm joints and gripper (normalized to 0-1)
            # Hardware gripper is 0-1000
            qpos = np.concatenate([joint_angles, [gripper / 1000.0]])
            
            return {
                'qpos': qpos,
                'joint_angles': joint_angles.copy(),
                'gripper': gripper,
                'timestamp': state.timestamp,
            }
        except Exception as e:
            logger.error(f"[AliciaD] get_observation error: {e}")
            return None
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode.
        
        For qpos mode: pass through (7D: 6 joint angles in rad + gripper 0-1)
        For delta_ee mode: convert 7D delta to joint positions using IK
        For rel_ee mode: convert 7D relative pose (from anchor) to joint positions using IK
        """
        if self.control_mode == "qpos":
            return action_dict
        
        # delta_ee and rel_ee modes
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        if len(action) != 7:
            logger.warning(f"[AliciaD] {self.control_mode} expects 7D action, got {len(action)}D")
            return action_dict
        
        # Parse action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        # Gripper delta is applied in start() before this call (delta_ee / rel_ee).
        pos_component = action[:3] * self.position_scale
        euler_component = action[3:6] * self.rotation_scale
        
        # Check if action is essentially zero
        if np.allclose(action, 0, atol=1e-6):
            # In rel_ee mode, we MUST process zero actions if they signal a state change
            # (start or stop) to keep the robot's anchor state synchronized with the teleop.
            if self.control_mode == "rel_ee":
                input_unsqueeze = action_dict.get('unsqueeze_active', True)
                anchor_set = action_dict.get('anchor_just_set', False)
                internal_unsqueeze = self._rel_ee_vr_unsqueeze_active
                
                # Pass through if:
                # 1. Anchor just set (Start of movement)
                # 2. Falling edge: Internal True -> Input False (End of movement)
                if anchor_set or (internal_unsqueeze and not input_unsqueeze):
                    pass # Continue to process_action logic (state change frame)
                else:
                    action_dict['action'] = None
                    return action_dict
            else:
                # For delta_ee/qpos, zero action means nothing to do
                action_dict['action'] = None
                return action_dict
        
        with self._lock:
            if self.control_mode == "delta_ee":
                # delta_ee mode: apply delta to IK solver's INTERNAL state (not actual joints)
                # This allows accumulation of deltas for smooth motion
                # Only sync with actual joints on first action or when explicitly requested
                
                # Initialize IK state if this is the first action
                if not hasattr(self, '_ik_initialized') or not self._ik_initialized:
                    if self._current_joint_angles is not None:
                        self.ik_solver.reset(self._current_joint_angles)
                        self._ik_initialized = True
                        logger.info(f"[AliciaD] delta_ee: IK initialized with actual joints (deg): {np.rad2deg(self._current_joint_angles)}")
                        logger.info(f"[AliciaD] delta_ee: initial EE pos: {self.ik_solver.get_current_pose()[:3, 3]}")
                    else:
                        logger.warning("[AliciaD] delta_ee: No current joint angles for IK init")
                        action_dict['action'] = None
                        return action_dict
                
                # Get current EE from IK solver's INTERNAL state (accumulated)
                current_T = self.ik_solver.get_current_pose()
                
                # Build target pose by applying delta to accumulated position
                target_T = current_T.copy()
                target_T[:3, 3] += pos_component
                
                # Convert euler delta to rotation matrix (base frame delta rotation)
                if np.linalg.norm(euler_component) > 1e-10:
                    from scipy.spatial.transform import Rotation as R
                    R_delta_base = R.from_euler('xyz', euler_component).as_matrix()
                    target_T[:3, :3] = R_delta_base @ current_T[:3, :3]
                
                # Gripper: _target_gripper already includes this frame's teleop delta (see start() loop).
                new_gripper = float(np.clip(self._target_gripper, 0, 1000))
                
            elif self.control_mode == "rel_ee":
                try:
                    # rel_ee mode: apply relative pose to ANCHOR
                    vr_unsqueeze_active = action_dict.get('unsqueeze_active', True)
                    anchor_just_set = action_dict.get('anchor_just_set', False)
                    
                    # Detect VR unsqueeze state transitions
                    vr_was_active = self._rel_ee_vr_unsqueeze_active
                    self._rel_ee_vr_unsqueeze_active = vr_unsqueeze_active
                    
                    if vr_unsqueeze_active and not vr_was_active and not anchor_just_set:
                        logger.warning("[AliciaD] rel_ee: unsqueeze rising edge but anchor_just_set=False (teleop may be missing press frame).")

                    if not vr_unsqueeze_active:
                        # If unsqueeze is released, just update state and hold position.
                        action_dict['action'] = None
                        return action_dict

                    # Determine trigger reason
                    trigger_reason = ""
                    if anchor_just_set: 
                        trigger_reason = "anchor_just_set"
                    elif (vr_unsqueeze_active and not vr_was_active): 
                        trigger_reason = "unsqueeze_edge"

                    # Set anchor when triggered (fresh squeeze = full reset)
                    if trigger_reason:
                        success = self.refresh_rel_ee_anchor(use_actual_joints=True)
                        if not success:
                            logger.warning("[AliciaD] rel_ee: Failed to refresh anchor. Aborting action.")
                            action_dict['action'] = None
                            return action_dict
                        # Clear offset on fresh anchor (VR and robot are fully synced)
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)
                        # Safety: do not publish motion on the press frame (zero action).
                        if anchor_just_set and np.allclose(pos_component, 0, atol=1e-8) and np.allclose(euler_component, 0, atol=1e-8):
                            action_dict['action'] = None
                            return action_dict
                    
                    # If no anchor yet, try to set it now
                    if self._rel_ee_anchor_pose is None:
                        success = self.refresh_rel_ee_anchor(use_actual_joints=True)
                        if not success:
                            logger.warning("[AliciaD] rel_ee: Failed to set initial anchor")
                            action_dict['action'] = None
                            return action_dict
                        self._rel_ee_pos_offset = np.zeros(3)
                        self._rel_ee_rot_offset = np.zeros(3)
                    
                    # Apply offset correction: subtract the accumulated offset from VR's
                    # relative pose so that the robot's target stays within reachable space.
                    corrected_pos = pos_component - self._rel_ee_pos_offset
                    corrected_euler = euler_component - self._rel_ee_rot_offset
                    
                    # Build target pose by applying corrected relative transform to anchor
                    target_T = self._rel_ee_anchor_pose.copy()
                    target_T[:3, 3] += corrected_pos
                    
                    # Apply relative rotation (in base frame)
                    if np.linalg.norm(corrected_euler) > 1e-10:
                        from scipy.spatial.transform import Rotation as R
                        R_rel_base = R.from_euler('xyz', corrected_euler).as_matrix()
                        target_T[:3, :3] = R_rel_base @ self._rel_ee_anchor_pose[:3, :3]
                    
                    # Save raw (uncorrected) VR relative pose for potential offset absorption
                    self._rel_ee_last_raw_pos = pos_component.copy()
                    self._rel_ee_last_raw_euler = euler_component.copy()
                    
                    # Gripper: _target_gripper already includes this frame's teleop delta (see start() loop).
                    new_gripper = float(np.clip(self._target_gripper, 0, 1000))
                    
                except Exception as e:
                    logger.error(f"[AliciaD] rel_ee process error: {e}")
                    import traceback
                    traceback.print_exc()
                    action_dict['action'] = None
                    return action_dict
            
            else:
                return action_dict
            
            # Solve IK with target pose
            success, q_arm, iters, err = self.ik_solver.solve(target_T)
            
            # IK Recovery: If failed, try resetting to a known-good configuration and solve again.
            # Prefer _target_qpos (last commanded) over _current_joint_angles (encoder)
            # because the hardware encoder is unreliable and often reports near-zero.
            if not success:
                recovery_joints = self._target_qpos if self._target_qpos is not None else self._current_joint_angles
                if recovery_joints is not None:
                    self.ik_solver.reset(recovery_joints)
                    success, q_arm, iters, err = self.ik_solver.solve(target_T)
            
            # SAFETY: If IK failed, reject the action and dynamically re-anchor (rel_ee mode)
            if not success:
                if self.control_mode == "rel_ee" and hasattr(self, '_rel_ee_last_raw_pos'):
                    # Dynamic re-anchor: absorb the current VR relative pose into offset
                    # so that subsequent frames start from the last reachable position.
                    self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                    self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                    # Refresh robot anchor to last successfully commanded pose
                    self._reanchor_to_last_good_pose()
                    self._warning_throttled_ik(
                        f"[AliciaD] IK failed, re-anchored: err={err:.4e}, target_pos={target_T[:3,3]}"
                    )
                else:
                    self._warning_throttled_ik(
                        f"[AliciaD] IK failed (unreachable target): err={err:.4e}, iters={iters}, target_pos={target_T[:3,3]}"
                    )
                action_dict['action'] = None
                return action_dict
            
            # SAFETY: Reject IK solution if it jumps too far from last known position.
            # This catches "flip solutions" where IK converges to a valid but distant config.
            reference_q = self._target_qpos if self._target_qpos is not None else self._current_joint_angles
            if reference_q is not None:
                ik_jump = np.max(np.abs(q_arm - reference_q))
                if ik_jump > self.max_joint_delta_rad:
                    if self.control_mode == "rel_ee" and hasattr(self, '_rel_ee_last_raw_pos'):
                        self._rel_ee_pos_offset = self._rel_ee_last_raw_pos.copy()
                        self._rel_ee_rot_offset = self._rel_ee_last_raw_euler.copy()
                        self._reanchor_to_last_good_pose()
                    else:
                        self.ik_solver.reset(reference_q)
                    self._warning_throttled_ik(
                        f"[AliciaD] IK solution REJECTED (jump): max joint jump {np.rad2deg(ik_jump):.1f}° "
                        f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                    )
                    action_dict['action'] = None
                    return action_dict
            
            # IK succeeded: clear any accumulated offset drift (successful convergence
            # means robot and VR are in sync at this corrected pose)
            # NOTE: We do NOT clear offset here. The offset stays until the user
            # re-squeezes (full anchor reset). This ensures continuity.
            
            # Store targets for next iteration
            self._target_qpos = q_arm.copy()
            self._target_gripper = new_gripper
            
            # Build output action: [6 joint angles, gripper_normalized]
            output_action = np.concatenate([q_arm, [new_gripper / 1000.0]])
            action_dict['action'] = output_action
        
        return action_dict
    
    def _reanchor_to_last_good_pose(self):
        """
        Re-anchor to the last successfully commanded pose (IK failure recovery).
        
        This refreshes the robot-side anchor to the last known reachable EE pose,
        so that subsequent VR relative poses (after offset correction) will target
        positions near the robot's actual position instead of the unreachable one.
        """
        joints = self._target_qpos
        if joints is not None:
            fk_pose = self.ik_solver.compute_fk(joints)
            self._rel_ee_anchor_pose = fk_pose.copy()
            self.ik_solver.reset(joints)
            self.ik_solver.set_reference(joints, fk_pose)
    
    def _warning_throttled_ik(self, msg: str):
        """Throttled IK warning to avoid spamming at 200Hz."""
        now = time.time()
        if not hasattr(self, '_last_ik_warn_time'):
            self._last_ik_warn_time = 0.0
            self._ik_warn_suppressed = 0
        if now - self._last_ik_warn_time >= 1.0:
            if self._ik_warn_suppressed > 0:
                msg += f" (suppressed {self._ik_warn_suppressed} repeats)"
            logger.warning(msg)
            self._last_ik_warn_time = now
            self._ik_warn_suppressed = 0
        else:
            self._ik_warn_suppressed += 1
    
    def refresh_rel_ee_anchor(self, use_actual_joints: bool = True) -> bool:
        """
        Refresh the rel_ee anchor to current EE pose.
        
        The anchor is computed using forward kinematics from actual joint observations.
        """
        if self.control_mode != "rel_ee":
            return False
        
        with self._lock:
            # FORCE UPDATE: Get fresh observation immediately
            self.get_observation()
            
            # Determine which joint source to use for anchor FK calculation.
            #
            # IMPORTANT: This robot's hardware encoder does NOT reliably track 
            # commanded positions - it often reports near-zero (home) values 
            # even after the robot has moved. Therefore we PREFER _target_qpos 
            # (the last successfully commanded joints) over _current_joint_angles 
            # (hardware encoder readout).
            #
            # Priority:
            # 1. _target_qpos (last commanded) - most reliable
            # 2. _current_joint_angles (hardware encoder) - fallback only
            joints_to_use = None
            
            if self._target_qpos is not None:
                joints_to_use = self._target_qpos
            elif self._current_joint_angles is not None:
                joints_to_use = self._current_joint_angles
            
            if joints_to_use is not None:
                fk_pose = self.ik_solver.compute_fk(joints_to_use)
                self._rel_ee_anchor_pose = fk_pose.copy()
                self.ik_solver.reset(joints_to_use)
                self.ik_solver.set_reference(joints_to_use, fk_pose)
            else:
                logger.warning("[AliciaD] refresh_anchor failed: No joints available")
                return False
            
            self._rel_ee_anchor_gripper = self._current_gripper if self._current_gripper is not None else self._target_gripper
            self._rel_ee_anchor_initialized = True
            
            return True
    
    def publish_action(self, action: np.ndarray):
        """
        Publish action to robot with safety checks.
        
        Args:
            action: 7D array [6 joint angles in rad, gripper normalized 0-1]
        """
        if not self._connected or self._driver is None:
            logger.error("[AliciaD] Not connected")
            return
        
        try:
            if len(action) < 7:
                logger.error(f"[AliciaD] Invalid action dimension: {len(action)}")
                return
            
            target_joints = np.array(action[:6], dtype=np.float64)
            # Convert gripper from 0-1 to 0-1000
            gripper_value = int(action[6] * 1000)
            gripper_value = np.clip(gripper_value, 0, 1000)
            
            # Safety check: reject if joint delta is too large.
            # Prefer _target_qpos (last commanded) over _current_joint_angles (encoder)
            # because the hardware encoder is unreliable and often reports near-zero.
            reference_joints = self._target_qpos if self._target_qpos is not None else self._current_joint_angles
            if reference_joints is not None:
                joint_delta = np.abs(target_joints - reference_joints)
                max_delta = np.max(joint_delta)
                
                if max_delta > self.max_joint_delta_rad:
                    logger.warning(
                        f"[AliciaD] Action REJECTED: max joint delta {np.rad2deg(max_delta):.1f}° "
                        f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                    )
                    return
            else:
                logger.warning("[AliciaD] No reference joint angles available for safety check, skipping action")
                return
            
            # Send to hardware via driver
            # The driver.set_joint_and_gripper method is unified and non-blocking by default unless wait_for_completion=True
            self._driver.set_joint_and_gripper(
                joint_angles=target_joints.tolist(),
                gripper_value=gripper_value,
                speed_deg_s=self.speed_deg_s,
                gripper_speed_deg_s=self.gripper_speed_deg_s
            )
            
            # Update target tracking
            self._target_qpos = target_joints
            self._target_gripper = gripper_value
            
        except Exception as e:
            logger.error(f"[AliciaD] publish_action error: {e}")
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return 7  # 6 arm joints + 1 gripper
    
    def is_running(self) -> bool:
        """Check if robot is running."""
        # Check if driver is connected and update thread is running
        if self._driver is None:
            return False
        return self._connected and self._driver.serial_comm.is_connected()
    
    def shutdown(self):
        """Shutdown robot."""
        try:
            if self._driver is not None:
                self._driver.disconnect()
                self._driver = None
            self._connected = False
            logger.info("[AliciaD] Shutdown complete")
        except Exception as e:
            logger.error(f"[AliciaD] Shutdown error: {e}")
    
    def start(self):
        """
        Start the robot in subprocess.
        
        IMPORTANT: Connection must be established in the subprocess, not in the main process.
        This is because serial connections cannot be inherited across process boundaries
        (especially with 'spawn' start method).
        """
        from deploy.utils import RateLimiter
        
        # Create shared memory for observations
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Connect to control shm (teleop) with retry - it may not be ready yet
        if self.control_shm_name is not None and self.control_shm is None:
            max_retries = 10
            for i in range(max_retries):
                try:
                    self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    logger.info(f"Connected to control SHM: {self.control_shm_name}")
                    break
                except ValueError as e:
                    if i < max_retries - 1:
                        logger.warning(f"Waiting for control SHM '{self.control_shm_name}'... ({i+1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to connect to control SHM: {e}")
        
        # IMPORTANT: Connect to robot hardware in subprocess
        # Serial connections cannot be inherited from main process
        if not self._connected:
            logger.info("[AliciaD] Connecting to robot in subprocess...")
            if not self.connect():
                logger.error("[AliciaD] Failed to connect to robot in subprocess")
                return
        
        self.is_running = True
        rate_limiter = RateLimiter()
        
        logger.info(
            f"[AliciaD] Starting main loop (mode={self.control_mode}, fps={self.fps})"
        )
        if self.gripper_trace:
            logger.warning(
                "[AliciaD] gripper_trace=ON: logging [GripperTrace:robot] lines; "
                "disable via gripper_trace: false in robot YAML after debugging."
            )
        
        # Idle counter for auto-resetting unsqueeze state
        _action_idle_count = 0
        _trace_prev_unsqueeze: Optional[bool] = None
        
        while self.is_running:
            # Read and process action from teleop (arrives at teleop rate, e.g. 50Hz)
            action = self.read_action()
            if action is not None:
                _action_idle_count = 0
                
                # Check for go_home command (e.g. B button on VR controller)
                if action.get('go_home', False):
                    logger.info("[AliciaD] Go-home command received, moving to home position...")
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] GO_HOME rx: tgt_pre={self._target_gripper:.1f} "
                            f"cur_hw={self._current_gripper}"
                        )
                    self.set_home(speed_deg_s=self.speed_deg_s)
                    # Reset IK / anchor state so next movement starts fresh
                    self._ik_initialized = False
                    self._rel_ee_anchor_pose = None
                    self._rel_ee_vr_unsqueeze_active = False
                    self._rel_ee_pos_offset = np.zeros(3)
                    self._rel_ee_rot_offset = np.zeros(3)
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] GO_HOME after set_home: tgt={self._target_gripper:.1f} "
                            f"tq0={self._target_qpos[:3] if self._target_qpos is not None else None}"
                        )
                    continue
                
                unsqz = bool(action.get("unsqueeze_active", False))
                if self.gripper_trace and _trace_prev_unsqueeze is not None and unsqz != _trace_prev_unsqueeze:
                    edge = "UNFREEZE" if unsqz else "FREEZE"
                    ra6 = None
                    raw = action.get("action")
                    if raw is not None and len(raw) >= 7:
                        ra6 = float(raw[6])
                    logger.info(
                        f"[GripperTrace:robot] {edge}: tgt={self._target_gripper:.1f} raw_a6={ra6} "
                        f"go_home={action.get('go_home')} anchor_set={action.get('anchor_just_set')}"
                    )
                _trace_prev_unsqueeze = unsqz

                # Extract gripper delta BEFORE process_action, so gripper works
                # independently of IK success/failure and unsqueeze state.
                # (delta_ee / rel_ee only — qpos passes absolute gripper in process_action path.)
                raw_action = action.get('action', None)
                tg_before = float(self._target_gripper)
                if (
                    self.control_mode in ("delta_ee", "rel_ee")
                    and raw_action is not None
                    and len(raw_action) >= 7
                ):
                    gripper_delta = float(raw_action[6]) * self.gripper_scale
                    self._target_gripper = float(np.clip(
                        self._target_gripper + gripper_delta * 1000, 0, 1000
                    ))
                    if self.gripper_trace and (
                        abs(gripper_delta) > 1e-9 or abs(self._target_gripper - tg_before) > 0.5
                    ):
                        logger.info(
                            f"[GripperTrace:robot] accum: raw6={float(raw_action[6]):.5f} "
                            f"scale={self.gripper_scale} d_scaled={gripper_delta:.5f} "
                            f"tgt {tg_before:.1f}->{self._target_gripper:.1f}"
                        )
                
                action = self.process_action(action)
                action_array = action.get('action', None)
                if action_array is not None:
                    out = np.asarray(action_array, dtype=np.float64).copy()
                    if self.control_mode == "qpos":
                        pass  # gripper already absolute in teleop command
                    else:
                        out[6] = self._target_gripper / 1000.0
                    if self.gripper_trace:
                        logger.info(
                            f"[GripperTrace:robot] publish: out6={out[6]:.4f} tgt_raw={self._target_gripper:.1f} "
                            f"unsqz={action.get('unsqueeze_active')}"
                        )
                    self.publish_action(out)
                elif self.gripper_trace and self.control_mode in ("delta_ee", "rel_ee"):
                    ra6 = float(raw_action[6]) if raw_action is not None and len(raw_action) >= 7 else None
                    logger.info(
                        f"[GripperTrace:robot] no_publish (IK/skip): tgt={self._target_gripper:.1f} raw6={ra6}"
                    )
            else:
                _action_idle_count += 1
                # If idle for >0.1s (20 frames at 200Hz), assume teleop stopped sending
                if _action_idle_count > 20 and self._rel_ee_vr_unsqueeze_active:
                    self._rel_ee_vr_unsqueeze_active = False
            
            # Get observation and write to shm
            data = self.get_data()
            if data is not None:
                self.write_data(data)
            
            rate_limiter.sleep(self.fps)
    
    def obs2meta(self, device_data: dict) -> dict:
        """Extract robot state from this device's SHM data."""
        if device_data is None:
            return {}
        qpos = device_data.get('qpos', np.zeros(7, dtype=np.float32))
        return {'state': np.asarray(qpos, dtype=np.float32)}
    
    def meta2act(self, mact) -> np.ndarray:
        """Convert MetaAction to robot action."""
        return mact.get('action', np.zeros(7))
    
    # ==================== Convenience Methods ====================
    
    def set_home(self, speed_deg_s: float = 10.0):
        """Move robot to home position (all zeros) with gripper fully open (1000)."""
        if self._driver is not None:
            home_joints = [0.0] * 6
            gripper_open = 1000
            self._driver.set_joint_and_gripper(
                joint_angles=home_joints,
                gripper_value=gripper_open,
                speed_deg_s=speed_deg_s,
            )
            # Keep command state in sync — otherwise the next publish_action uses a stale
            # _target_gripper (e.g. pre-home closed) and immediately re-clamps the gripper
            # even when the teleop sends zero gripper delta.
            self._target_qpos = np.zeros(6, dtype=np.float64)
            self._target_gripper = float(gripper_open)
            self._current_gripper = float(gripper_open)
    
    def set_joints(self, joint_angles: np.ndarray, gripper: Optional[float] = None):
        """
        Set joint angles directly.
        
        Args:
            joint_angles: 6D array of joint angles in radians
            gripper: Gripper value (0-1), or None to keep current
        """
        if gripper is None:
            gripper = self._target_gripper / 1000.0 if self._target_gripper else 0.5
        
        action = np.concatenate([joint_angles, [gripper]])
        self.publish_action(action)
    
    def set_pose(self, target_pose: np.ndarray, gripper: Optional[float] = None) -> bool:
        """
        Set end-effector pose using IK.
        
        Args:
            target_pose: 4x4 transformation matrix
            gripper: Gripper value (0-1), or None to keep current
            
        Returns:
            True if IK succeeded
        """
        success, q_arm, iters, err = self.ik_solver.solve(target_pose)
        
        if gripper is None:
            gripper = self._target_gripper / 1000.0 if self._target_gripper else 0.5
        
        action = np.concatenate([q_arm, [gripper]])
        self.publish_action(action)
        
        return success
    
    def get_pose(self) -> np.ndarray:
        """Get current end-effector pose as 4x4 matrix."""
        if self._current_joint_angles is not None:
            return self.ik_solver.compute_fk(self._current_joint_angles)
        return self.ik_solver.get_current_pose()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Alicia-D Robot")
    parser.add_argument('--port', type=str, default='', help='Serial port (empty for auto-detect)')
    parser.add_argument('--mode', type=str, default='qpos', choices=['qpos', 'delta_ee', 'rel_ee'], help='Control mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    print(f"Creating Alicia-D robot on port '{args.port}' with mode={args.mode}")
    
    robot = AliciaD(
        port=args.port,
        control_mode=args.mode,
        debug=args.debug,
    )
    
    if not robot.connect():
        print("Failed to connect to robot")
        exit(1)
    
    print("Connected! Reading state...")
    
    try:
        for i in range(10):
            obs = robot.get_observation()
            if obs is not None:
                qpos = obs['qpos']
                print(f"[{i}] joints (deg): {np.rad2deg(qpos[:6])}, gripper: {qpos[6]:.3f}")
                
                # Get EE pose
                pose = robot.get_pose()
                print(f"     EE pos: {pose[:3, 3]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.shutdown()
        print("Done.")
