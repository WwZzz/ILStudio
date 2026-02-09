"""
Alicia-D Robot for ILStudio

Integrates Alicia-D robot arm with ILStudio framework using:
- Official alicia_d_sdk for hardware communication (via pip install)
- Custom Pinocchio-based real-time IK solver (fast, ~0.3ms per solve)
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


# ============================================================================
# Real-time Pinocchio IK Solver
# ============================================================================

class RealtimeIKSolver:
    """
    Optimized Pinocchio-based IK solver for real-time VR teleoperation.
    
    Key optimizations:
    - Warm start: uses previous joint configuration as initial guess
    - Reduced iterations: small pose changes converge quickly
    - Large step size: faster convergence for incremental motion
    - Caches model/data to avoid repeated initialization
    
    Typical performance: 0.1-0.5ms per solve (>2000Hz capable)
    """
    
    def __init__(
        self, 
        urdf_path: str, 
        end_frame: str = "tool0",
        arm_dof: int = 6,
        eps: float = 1e-3,
        max_iter: int = 20,
        dt: float = 1.0,
        damp: float = 1e-6,
        rotation_weight: float = 1.0,
    ):
        import pinocchio as pin
        
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(end_frame)
        
        self.arm_dof = arm_dof
        self.eps = eps
        self.max_iter = max_iter
        self.dt = dt
        self.damp = damp
        
        # Task-space weighting: position vs rotation priority.
        #
        # Pinocchio LOCAL frame convention for 6D spatial vectors:
        #   pin.log(SE3).vector  = [angular(3), linear(3)]
        #   computeFrameJacobian = [angular(3); linear(3)]  (row order)
        #
        # rotation_weight < 1.0 makes IK prioritize position accuracy over rotation,
        # reducing "wrist whipping" when precise rotation alignment would require
        # large joint movements.
        self.rotation_weight = rotation_weight
        # Pre-compute diagonal weight vector: [w_rot, w_rot, w_rot, 1, 1, 1]
        self._task_weight = np.array([rotation_weight] * 3 + [1.0] * 3)
        
        # Warm start state
        self._q_current = pin.neutral(self.model)
        
        # Pre-allocate arrays for speed
        self._v = np.zeros(self.model.nv)
        self._eye = np.eye(arm_dof)
        
    def reset(self, q: Optional[np.ndarray] = None):
        """Reset warm start state to given or neutral configuration."""
        if q is not None:
            # Ensure q has full DOF (pad with zeros for gripper if needed)
            if len(q) < self.model.nq:
                q_full = self.pin.neutral(self.model)
                q_full[:len(q)] = q
                self._q_current = q_full
            else:
                self._q_current = q.copy()
        else:
            self._q_current = self.pin.neutral(self.model)
    
    def get_current_pose(self) -> np.ndarray:
        """Get current end-effector pose as 4x4 matrix (from internal IK state)."""
        self.pin.forwardKinematics(self.model, self.data, self._q_current)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        T = np.eye(4)
        T[:3, :3] = pose.rotation
        T[:3, 3] = pose.translation
        return T
    
    def compute_fk(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint angles.
        
        This is used to get the actual EE pose from observed joint angles,
        independent of the IK solver's internal state.
        
        Args:
            joint_angles: Joint angles in radians (6D for arm)
            
        Returns:
            4x4 transformation matrix of end-effector pose
        """
        # Pad joint angles to full DOF if needed
        q = self.pin.neutral(self.model)
        q[:len(joint_angles)] = joint_angles
        
        # Compute FK
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.frame_id]
        
        T = np.eye(4)
        T[:3, :3] = pose.rotation
        T[:3, 3] = pose.translation
        return T
    
    def get_current_q(self) -> np.ndarray:
        """Get current joint configuration (arm joints only)."""
        return self._q_current[:self.arm_dof].copy()
    
    def solve(self, target_pose: np.ndarray) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve IK for target pose using warm start with weighted damped least squares.
        
        Position and rotation are weighted differently via self._task_weight,
        so that rotation_weight < 1.0 makes the solver prioritize position accuracy
        and avoids large joint movements just to align rotation precisely.
        
        Args:
            target_pose: 4x4 transformation matrix
            
        Returns:
            Tuple of (success, q_arm, iterations, error_norm)
            
        SAFETY: If IK fails, internal state is NOT updated to prevent cascading errors.
        """
        pin = self.pin
        target = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        q = self._q_current.copy()
        w = self._task_weight  # [w_rot, w_rot, w_rot, 1, 1, 1]
        
        for i in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            dMi = self.data.oMf[self.frame_id].actInv(target)
            err = pin.log(dMi).vector
            
            # Weighted error: scale rotation components down
            err_w = w * err
            err_norm = np.linalg.norm(err_w)
            
            if err_norm < self.eps:
                # SUCCESS: update state and return
                self._q_current = q
                return True, q[:self.arm_dof].copy(), i + 1, err_norm
            
            # Compute Jacobian (only arm joints)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, 
                pin.ReferenceFrame.LOCAL
            )[:, :self.arm_dof]
            
            # Weighted Jacobian: scale rotation rows down
            # This is equivalent to: J_w = diag(w) @ J
            J_w = w[:, None] * J
            
            # Weighted damped least squares:
            # min ||W(J*v - e)||^2 + damp*||v||^2
            # => v = (J_w^T J_w + damp*I)^{-1} J_w^T e_w
            v_arm = np.linalg.solve(J_w.T @ J_w + self._eye * self.damp, J_w.T @ err_w)
            
            # Update
            self._v[:self.arm_dof] = v_arm
            self._v[self.arm_dof:] = 0
            q = pin.integrate(self.model, q, self._v * self.dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        
        # FAILED: DO NOT update _q_current - keep previous valid state
        # Return the failed result but caller should NOT execute it
        return False, q[:self.arm_dof].copy(), self.max_iter, err_norm
    
    def solve_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve IK for incremental motion (common in VR teleoperation).
        
        Args:
            delta_pos: Position delta [dx, dy, dz] in meters
            delta_rot: Rotation delta as axis-angle [rx, ry, rz] in radians
            
        Returns:
            Tuple of (success, q_arm, iterations, error_norm)
        """
        # Get current pose
        current_T = self.get_current_pose()
        
        # Apply delta
        target_T = current_T.copy()
        target_T[:3, 3] += delta_pos
        if np.linalg.norm(delta_rot) > 1e-10:
            delta_R = self.pin.exp3(delta_rot)
            target_T[:3, :3] = current_T[:3, :3] @ delta_R
        
        return self.solve(target_T)


# ============================================================================
# Action Smoother (Joint-space Low-pass Filter)
# ============================================================================

class ActionSmoother:
    """
    Time-aware exponential low-pass filter for joint-space trajectories.
    
    Produces smooth, continuous motion at the robot's full control rate (e.g. 200Hz),
    even when IK targets arrive at a much lower rate (e.g. teleop at 50Hz).
    
    Uses first-order exponential smoothing with a configurable time constant (tau):
        q_smooth += alpha * (q_target - q_smooth)
        alpha = 1 - exp(-dt / tau)
    
    Properties:
    - Frame-rate independent: same physical behavior regardless of loop frequency
    - Zero overshoot: always converges monotonically toward target
    - 63% convergence in tau seconds, 95% in ~3*tau seconds
    - Negligible computational cost (~1us per call)
    
    Recommended tau values:
    - 0.02s: Light smoothing, very responsive (good for skilled teleop)
    - 0.04s: Medium smoothing, balanced (recommended starting point)
    - 0.08s: Heavy smoothing, very smooth but noticeable lag
    """
    
    def __init__(self, ndof: int = 7, tau: float = 0.04):
        """
        Args:
            ndof: Number of degrees of freedom (6 joints + 1 gripper)
            tau: Smoothing time constant in seconds. Larger = smoother but laggier.
                 Set to 0 to disable smoothing (pass-through).
        """
        self.ndof = ndof
        self.tau = tau
        self._target = None    # Latest IK target
        self._current = None   # Current smoothed output
        self._last_time = None
    
    def set_target(self, target: np.ndarray):
        """Set a new target from IK solution."""
        target = np.asarray(target, dtype=np.float64)
        if self._current is None:
            # First target ever: snap to it immediately (no smoothing)
            self._current = target.copy()
            self._last_time = time.perf_counter()
        self._target = target.copy()
    
    def hold(self):
        """
        Hold at current smoothed position.
        Call when control is paused (e.g. unsqueeze released) to prevent
        the smoother from continuing to drift toward a stale target.
        """
        if self._current is not None:
            self._target = self._current.copy()
    
    def reset(self):
        """Reset all state. Next set_target() will snap."""
        self._target = None
        self._current = None
        self._last_time = None
    
    def step(self) -> Optional[np.ndarray]:
        """
        Advance one smoothing step. Call every loop iteration.
        
        Returns:
            Smoothed action array, or None if no target has been set.
        """
        if self._target is None or self._current is None:
            return None
        
        now = time.perf_counter()
        dt = now - self._last_time if self._last_time is not None else 0.0
        self._last_time = now
        
        # Bypass smoothing if tau is zero or negligible
        if self.tau <= 1e-6 or dt <= 0:
            self._current = self._target.copy()
            return self._current.copy()
        
        # Exponential smoothing: alpha = 1 - exp(-dt / tau)
        # At 200Hz (dt=5ms) with tau=0.04: alpha ≈ 0.118 → smooth convergence
        alpha = 1.0 - np.exp(-dt / self.tau)
        self._current += alpha * (self._target - self._current)
        return self._current.copy()
    
    @property
    def has_target(self) -> bool:
        return self._target is not None


# ============================================================================
# Alicia-D Robot Class
# ============================================================================

class AliciaD(BaseRobot):
    """
    Alicia-D Robot for ILStudio
    
    Uses alicia_d_sdk.hardware for low-level communication.
    
    Supports three control modes:
    1. "qpos" (default): Direct joint position control (7D: 6 arm + 1 gripper)
    2. "delta_ee": 7D delta EE control with fast Pinocchio IK
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
        # IK settings
        ik_eps: float = 2e-3,    # Relaxed to 2mm for better stability
        ik_max_iter: int = 100,  # Increased from 20 to 100
        ik_dt: float = 0.5,      # Reduced from 1.0 to reduce oscillation
        ik_damp: float = 2e-2,   # Increased damping (0.02) to prevent wrist whipping near singularities
        ik_rotation_weight: float = 0.3,  # Rotation weight in IK (< 1.0 prioritizes position over rotation)
        # Delta EE scaling
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        # Speed settings
        speed_deg_s: float = 30.0,
        gripper_speed_deg_s: float = 483.4,
        # Smoothing
        smoothing_tau: float = 0.04,  # Joint trajectory smoothing time constant (seconds). 0 = disabled.
        # Safety settings
        max_joint_delta_deg: float = 5.0,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize Alicia-D Robot.
        
        Args:
            name: Name for shared memory
            max_size_mb: Maximum shared memory size in MB
            fps: Control frequency in Hz
            control_shm_name: Name of control shared memory (for receiving actions)
            port: Serial port for robot (empty string for auto-detect)
            urdf_path: Path to URDF file (auto-detected if None)
            control_mode: "qpos" for joint control, "delta_ee" or "rel_ee" for IK-based EE control
            gripper_type: Gripper type ("50mm" or "100mm")
            ik_eps: IK convergence tolerance (meters)
            ik_max_iter: Maximum IK iterations
            ik_dt: IK step size
            position_scale: Scale for position deltas in delta_ee/rel_ee mode
            rotation_scale: Scale for rotation deltas in delta_ee/rel_ee mode
            gripper_scale: Scale for gripper deltas in delta_ee/rel_ee mode
            speed_deg_s: Default arm speed in deg/s
            gripper_speed_deg_s: Default gripper speed in deg/s
            max_joint_delta_deg: Safety limit - max joint change per action in degrees
            debug: Enable debug output
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.port = port
        self.debug = debug
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
        
        # Smoothing
        self.smoothing_tau = smoothing_tau
        
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
        
        # IK solver (lazy initialized)
        self._ik_solver: Optional[RealtimeIKSolver] = None
        self._ik_settings = {
            'eps': ik_eps,
            'max_iter': ik_max_iter,
            'dt': ik_dt,
            'damp': ik_damp,
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
        
        # Connection state
        self._connected = False
        self.is_running = False
        
        logger.info(f"[AliciaD] Initialized with control_mode={control_mode}, fps={fps}")
        if control_mode in ("delta_ee", "rel_ee"):
            logger.info(f"[AliciaD] Delta EE scales: pos={position_scale}, rot={rotation_scale}, gripper={gripper_scale}")
    
    @property
    def ik_solver(self) -> RealtimeIKSolver:
        """Lazy-initialize IK solver."""
        if self._ik_solver is None:
            self._ik_solver = RealtimeIKSolver(
                urdf_path=self.urdf_path,
                end_frame="link6",
                arm_dof=6,
                eps=self._ik_settings['eps'],
                max_iter=self._ik_settings['max_iter'],
                dt=self._ik_settings['dt'],
                rotation_weight=self._ik_settings['rotation_weight'],
            )
            logger.info(
                f"[AliciaD] IK solver initialized: eps={self._ik_settings['eps']}, "
                f"max_iter={self._ik_settings['max_iter']}, rotation_weight={self._ik_settings['rotation_weight']}"
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
        pos_component = action[:3] * self.position_scale
        euler_component = action[3:6] * self.rotation_scale
        gripper_component = action[6] * self.gripper_scale
        
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
                
                # Gripper: accumulate delta
                new_gripper = np.clip(self._target_gripper + gripper_component * 1000, 0, 1000)
                
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

                    # Set anchor when triggered
                    if trigger_reason:
                        success = self.refresh_rel_ee_anchor(use_actual_joints=True)
                        if not success:
                            logger.warning("[AliciaD] rel_ee: Failed to refresh anchor. Aborting action.")
                            action_dict['action'] = None
                            return action_dict
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
                    
                    # Build target pose by applying relative transform to anchor
                    target_T = self._rel_ee_anchor_pose.copy()
                    target_T[:3, 3] += pos_component
                    
                    # Apply relative rotation (in base frame)
                    if np.linalg.norm(euler_component) > 1e-10:
                        from scipy.spatial.transform import Rotation as R
                        R_rel_base = R.from_euler('xyz', euler_component).as_matrix()
                        target_T[:3, :3] = R_rel_base @ self._rel_ee_anchor_pose[:3, :3]
                    
                    # Gripper: accumulate delta (same as delta_ee mode)
                    new_gripper = np.clip(self._target_gripper + gripper_component * 1000, 0, 1000)
                    
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
            
            # IK Recovery: If failed, try resetting to actual joints and solve again
            if not success and self._current_joint_angles is not None:
                self.ik_solver.reset(self._current_joint_angles)
                success, q_arm, iters, err = self.ik_solver.solve(target_T)
            
            # SAFETY: If IK failed, reject the action entirely
            if not success:
                logger.warning(f"[AliciaD] IK failed (unreachable target): err={err:.4e}, iters={iters}, target_pos={target_T[:3,3]}")
                action_dict['action'] = None
                return action_dict
            
            # Store targets for next iteration
            self._target_qpos = q_arm.copy()
            self._target_gripper = new_gripper
            
            # Build output action: [6 joint angles, gripper_normalized]
            output_action = np.concatenate([q_arm, [new_gripper / 1000.0]])
            action_dict['action'] = output_action
        
        return action_dict
    
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
            
            # Safety check: reject if joint delta is too large
            if self._current_joint_angles is not None:
                joint_delta = np.abs(target_joints - self._current_joint_angles)
                max_delta = np.max(joint_delta)
                
                if max_delta > self.max_joint_delta_rad:
                    logger.warning(
                        f"[AliciaD] Action REJECTED: max joint delta {np.rad2deg(max_delta):.1f}° "
                        f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                    )
                    return
            else:
                logger.warning("[AliciaD] No current joint angles available for safety check")
            
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
        
        # Action smoother: smooths IK output for continuous, jerk-free motion
        smoother = ActionSmoother(ndof=7, tau=self.smoothing_tau)
        
        logger.info(
            f"[AliciaD] Starting main loop (mode={self.control_mode}, fps={self.fps}, "
            f"smoothing_tau={self.smoothing_tau}s)"
        )
        
        # Idle counter for auto-resetting unsqueeze state
        _action_idle_count = 0
        
        while self.is_running:
            # Read and process action from teleop (arrives at teleop rate, e.g. 50Hz)
            action = self.read_action()
            if action is not None:
                _action_idle_count = 0
                action = self.process_action(action)
                action_array = action.get('action', None)
                if action_array is not None:
                    # New IK target: feed to smoother
                    smoother.set_target(action_array)
                else:
                    # process_action returned None (e.g. unsqueeze released, zero action)
                    # Hold smoother at current position to prevent drift
                    smoother.hold()
            else:
                _action_idle_count += 1
                # If idle for >0.1s (20 frames at 200Hz), assume teleop stopped sending
                if _action_idle_count > 20 and self._rel_ee_vr_unsqueeze_active:
                    self._rel_ee_vr_unsqueeze_active = False
            
            # Advance smoother and publish on EVERY iteration (200Hz),
            # even when no new IK target arrived. This interpolates between
            # sparse teleop frames for smooth, continuous motion.
            smoothed = smoother.step()
            if smoothed is not None:
                self.publish_action(smoothed)
            
            # Get observation and write to shm
            data = self.get_data()
            if data is not None:
                self.write_data(data)
            
            rate_limiter.sleep(self.fps)
    
    def obs2meta(self, obs: dict) -> MetaObs:
        """Convert observations to MetaObs format."""
        if obs is None:
            return None
        
        qpos = obs.get('qpos', np.zeros(7))
        
        # Handle images if present
        images = []
        if 'image' in obs:
            images = obs['image']
            if isinstance(images, dict):
                images = np.stack([images[k] for k in sorted(images.keys())], axis=0)
            if images.ndim == 3:
                images = images[None]
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        else:
            images = np.zeros((1, 3, 480, 640), dtype=np.uint8)
        
        return MetaObs(state=qpos, image=images)
    
    def meta2act(self, mact) -> np.ndarray:
        """Convert MetaAction to robot action."""
        return mact.get('action', np.zeros(7))
    
    # ==================== Convenience Methods ====================
    
    def set_home(self, speed_deg_s: float = 10.0):
        """Move robot to home position (all zeros)."""
        if self._driver is not None:
            home_joints = [0.0] * 6
            # Use default gripper value (1000)
            self._driver.set_joint_and_gripper(
                joint_angles=home_joints, 
                gripper_value=1000,
                speed_deg_s=speed_deg_s
            )
    
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
