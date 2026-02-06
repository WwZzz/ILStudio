"""
Alicia-D Robot for ILStudio

Integrates Alicia-D robot arm with ILStudio framework using:
- Custom Pinocchio-based real-time IK solver (fast, ~0.3ms per solve)
- Direct ServoDriver for hardware control
- SharedMemory for observation/action communication

Control modes:
1. qpos: Direct joint position control (6D arm joints + 1D gripper)
2. delta_ee: 7D delta end-effector control with IK (dx, dy, dz, drx, dry, drz, dgripper)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Add alicia_d_sdk to path for its internal imports
_sdk_path = Path(__file__).parent / "alicia_d_sdk"
if str(_sdk_path.parent) not in sys.path:
    sys.path.insert(0, str(_sdk_path.parent))

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
        """Get current end-effector pose as 4x4 matrix."""
        self.pin.forwardKinematics(self.model, self.data, self._q_current)
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
        Solve IK for target pose using warm start.
        
        Args:
            target_pose: 4x4 transformation matrix
            
        Returns:
            Tuple of (success, q_arm, iterations, error_norm)
            
        SAFETY: If IK fails, internal state is NOT updated to prevent cascading errors.
        """
        pin = self.pin
        target = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        q = self._q_current.copy()
        
        for i in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            dMi = self.data.oMf[self.frame_id].actInv(target)
            err = pin.log(dMi).vector
            err_norm = np.linalg.norm(err)
            
            if err_norm < self.eps:
                # SUCCESS: update state and return
                self._q_current = q
                return True, q[:self.arm_dof].copy(), i + 1, err_norm
            
            # Compute Jacobian (only arm joints)
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, 
                pin.ReferenceFrame.LOCAL
            )[:, :self.arm_dof]
            
            # Damped least squares
            v_arm = np.linalg.solve(J.T @ J + self._eye * self.damp, J.T @ err)
            
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
# Alicia-D Robot Class
# ============================================================================

class AliciaD(BaseRobot):
    """
    Alicia-D Robot for ILStudio
    
    Supports two control modes:
    1. "qpos" (default): Direct joint position control (7D: 6 arm + 1 gripper)
    2. "delta_ee": 7D delta EE control with fast Pinocchio IK
    """
    
    # Control modes
    CONTROL_MODES = ("qpos", "delta_ee")
    
    def __init__(
        self,
        name: str = "alicia_d",
        max_size_mb: int = 64,
        fps: float = 200.0,
        control_shm_name: Optional[str] = None,
        port: str = "/dev/ttyACM0",
        urdf_path: Optional[str] = None,
        control_mode: str = "qpos",
        # IK settings
        ik_eps: float = 1e-3,
        ik_max_iter: int = 20,
        ik_dt: float = 1.0,
        # Delta EE scaling
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_scale: float = 1.0,
        # Speed settings
        speed_deg_s: float = 50.0,
        gripper_speed_deg_s: float = 483.4,
        # Safety settings
        max_joint_delta_deg: float = 30.0,  # Max allowed joint change per action (degrees)
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
            port: Serial port for robot
            urdf_path: Path to URDF file (auto-detected if None)
            control_mode: "qpos" for joint control, "delta_ee" for IK-based EE control
            ik_eps: IK convergence tolerance (meters)
            ik_max_iter: Maximum IK iterations
            ik_dt: IK step size
            position_scale: Scale for position deltas in delta_ee mode
            rotation_scale: Scale for rotation deltas in delta_ee mode
            gripper_scale: Scale for gripper deltas in delta_ee mode
            speed_deg_s: Default arm speed in deg/s
            gripper_speed_deg_s: Default gripper speed in deg/s
            max_joint_delta_deg: Safety limit - max joint change per action in degrees
            debug: Enable debug output
        """
        super().__init__(name=name, max_size_mb=max_size_mb, fps=fps, control_shm_name=control_shm_name)
        
        self.port = port
        self.debug = debug
        
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
        self.max_joint_delta_rad = np.deg2rad(max_joint_delta_deg)  # Convert to radians
        
        # Find URDF path - default to local file, no external dependency needed
        if urdf_path is None:
            # Use local URDF file (bundled with this module)
            local_urdf = Path(__file__).parent / "urdf" / "Alicia_D_v5_6_gripper_50mm.urdf"
            if local_urdf.exists():
                urdf_path = str(local_urdf)
            else:
                # Fallback: try synriard if available (optional dependency)
                try:
                    from synriard import get_model_path
                    urdf_path = str(get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm", model_format="urdf"))
                except ImportError:
                    raise FileNotFoundError(
                        f"URDF file not found at {local_urdf}. "
                        "Please provide urdf_path argument or install synriard package."
                    )
        self.urdf_path = urdf_path
        
        # Initialize hardware driver
        from alicia_d_sdk.hardware import ServoDriver
        self.servo_driver = ServoDriver(port=port, debug_mode=debug)
        
        # IK solver (lazy initialized)
        self._ik_solver: Optional[RealtimeIKSolver] = None
        self._ik_settings = {
            'eps': ik_eps,
            'max_iter': ik_max_iter,
            'dt': ik_dt,
        }
        
        # State tracking (for IK warm-start, not for caching obs)
        self._current_joint_angles: Optional[np.ndarray] = None
        self._current_gripper: Optional[float] = None
        self._target_qpos: Optional[np.ndarray] = None  # Target joint positions for IK
        self._target_gripper: float = 500.0
        self._lock = threading.Lock()
        
        # Connection state
        self._connected = False
        self.is_running = False
        
        logger.info(f"[AliciaD] Initialized with control_mode={control_mode}, fps={fps}")
        if control_mode == "delta_ee":
            logger.info(f"[AliciaD] Delta EE scales: pos={position_scale}, rot={rotation_scale}, gripper={gripper_scale}")
    
    @property
    def ik_solver(self) -> RealtimeIKSolver:
        """Lazy-initialize IK solver."""
        if self._ik_solver is None:
            self._ik_solver = RealtimeIKSolver(
                urdf_path=self.urdf_path,
                end_frame="tool0",
                arm_dof=6,
                eps=self._ik_settings['eps'],
                max_iter=self._ik_settings['max_iter'],
                dt=self._ik_settings['dt'],
            )
            logger.info(f"[AliciaD] IK solver initialized: eps={self._ik_settings['eps']}, max_iter={self._ik_settings['max_iter']}")
        return self._ik_solver
    
    def connect(self) -> bool:
        """Connect to robot."""
        if self._connected:
            return True
        
        try:
            # connect() starts the background serial read thread
            result = self.servo_driver.connect()
            if result:
                time.sleep(0.05)
                
                # Get initial state (blocking)
                state = self._get_fresh_state(timeout=1.0)
                if state is None:
                    logger.warning("[AliciaD] Initial state read failed")
                else:
                    self._current_joint_angles = np.array(state.angles, dtype=np.float64)
                    self._current_gripper = state.gripper
                    self._target_qpos = self._current_joint_angles.copy()
                    self._target_gripper = self._current_gripper
                
                # Initialize IK solver with current joint angles
                if self.control_mode == "delta_ee" and self._current_joint_angles is not None:
                    self.ik_solver.reset(self._current_joint_angles)
                
                self._connected = True
                logger.info(f"[AliciaD] Connected to robot on {self.port}")
                
                # Warning: first key press will move robot to zero position
                logger.info(
                    "[AliciaD] Press any key to move robot to ZERO position"
                    "Make sure the robot has clear space before starting."
                )
                
                return True
            else:
                logger.error(f"[AliciaD] Failed to connect to robot on {self.port}")
                return False
        except Exception as e:
            logger.error(f"[AliciaD] Connection error: {e}")
            return False
    
    def _get_fresh_state(self, timeout: float = 0.1):
        """
        Get FRESH robot state (synchronous, blocking).
        
        This is the ONLY way to get observation - no caching allowed.
        For imitation learning, we need real-time ground truth.
        """
        if not self.servo_driver.acquire_info("joint_gripper", wait=True, timeout=timeout):
            return None
        return self.servo_driver.data_parser.get_info("joint_gripper")
    
    def get_observation(self) -> Optional[Dict[str, Any]]:
        """
        Get complete observation data (ALWAYS fresh, no caching).
        
        For imitation learning data collection, observations must be
        real-time ground truth, not cached values.
        """
        try:
            # Always get fresh state - blocking but with short timeout
            state = self._get_fresh_state(timeout=0.1)
            if state is None:
                return None
            
            joint_angles = np.array(state.angles, dtype=np.float64)
            gripper = state.gripper
            
            # Update internal tracking
            self._current_joint_angles = joint_angles
            self._current_gripper = gripper
            
            # Combine arm joints and gripper (normalized to 0-1)
            qpos = np.concatenate([joint_angles, [gripper / 1000.0]])
            
            return {
                'qpos': qpos,
                'joint_angles': joint_angles.copy(),
                'gripper': gripper,
                'timestamp': time.time(),
            }
        except Exception as e:
            logger.error(f"[AliciaD] get_observation error: {e}")
            return None
    
    def process_action(self, action_dict: dict) -> dict:
        """
        Process action based on control mode.
        
        For qpos mode: pass through (7D: 6 joint angles in rad + gripper 0-1)
        For delta_ee mode: convert 7D delta to joint positions using IK
        
        NOTE: IK uses warm-start from previous target, not current obs.
        This allows IK to run independently of obs frequency.
        
        Rotation handling:
        - Input: xyz euler angles in robot base frame (from Quest3 after calibration_transform)
        - These are converted to a rotation matrix and applied as LEFT multiplication
        - target_R = R_delta_base @ current_R (rotation in base frame)
        
        Gripper control:
        - qpos mode: ABSOLUTE control, action[6] is normalized 0-1 (0=open, 1=closed)
                     Converted to hardware range 0-1000 (0=open, 1000=closed)
        - delta_ee mode: RELATIVE control, action[6] is delta value
                        Accumulated: new_gripper = clip(current + delta * 1000, 0, 1000)
                        Hardware range: 0-1000 (0=open, 1000=closed)
        """
        if self.control_mode == "qpos":
            return action_dict
        
        # delta_ee mode
        action = action_dict.get('action', None)
        if action is None:
            return action_dict
        
        action = np.array(action, dtype=np.float64)
        
        if len(action) != 7:
            logger.warning(f"[AliciaD] delta_ee expects 7D action, got {len(action)}D")
            return action_dict
        
        # Parse action: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        # Teleop (Quest3) outputs xyz euler angles in BASE frame after calibration_transform
        delta_pos = action[:3] * self.position_scale
        euler_delta = action[3:6] * self.rotation_scale  # xyz euler angles in base frame
        delta_gripper = action[6] * self.gripper_scale
        
        # Check if action is essentially zero
        if np.allclose(action, 0, atol=1e-6):
            action_dict['action'] = None
            return action_dict
        
        with self._lock:
            # Get current end-effector pose from IK solver's internal state
            current_T = self.ik_solver.get_current_pose()
            
            # Build target pose
            target_T = current_T.copy()
            target_T[:3, 3] += delta_pos  # Apply position delta
            
            # Convert euler delta to rotation matrix (base frame delta rotation)
            # Then LEFT multiply to current rotation (rotation in base/world frame)
            if np.linalg.norm(euler_delta) > 1e-10:
                from scipy.spatial.transform import Rotation as R
                R_delta_base = R.from_euler('xyz', euler_delta).as_matrix()
                # Left multiply: delta rotation is in base frame
                # target_R = R_delta_base @ current_R
                target_T[:3, :3] = R_delta_base @ current_T[:3, :3]
            
            # Solve IK with target pose (uses warm-start from previous solve)
            success, q_arm, iters, err = self.ik_solver.solve(target_T)
            
            # SAFETY: If IK failed, reject the action entirely
            # This prevents dangerous movements when target is unreachable
            if not success:
                if self.debug:
                    logger.warning(f"[AliciaD] IK failed (unreachable target): err={err:.4e}, iters={iters}")
                action_dict['action'] = None  # Reject action
                return action_dict
            
            # Update gripper target (only if IK succeeded)
            # RELATIVE control: delta_gripper is accumulated into current target
            # Range: 0-1000 (0=open, 1000=closed)
            new_gripper = np.clip(self._target_gripper + delta_gripper * 1000, 0, 1000)
            
            # Store targets for next iteration (only if IK succeeded)
            self._target_qpos = q_arm.copy()
            self._target_gripper = new_gripper
            
            # Build output action: [6 joint angles, gripper_normalized]
            output_action = np.concatenate([q_arm, [new_gripper / 1000.0]])
            action_dict['action'] = output_action
            
            if self.debug:
                logger.debug(f"[AliciaD] IK: iters={iters}, err={err:.4e}, q={np.rad2deg(q_arm)}")
        
        return action_dict
    
    def publish_action(self, action: np.ndarray):
        """
        Publish action to robot with safety checks.
        
        Args:
            action: 7D array [6 joint angles in rad, gripper normalized 0-1]
        
        Safety: Rejects action if joint delta exceeds max_joint_delta_rad
        """
        try:
            if len(action) < 7:
                logger.error(f"[AliciaD] Invalid action dimension: {len(action)}")
                return
            
            target_joints = np.array(action[:6], dtype=np.float64)
            # ABSOLUTE control: action[6] is normalized 0-1, convert to hardware range 0-1000
            # Hardware range: 0=open, 1000=closed
            gripper_value = int(action[6] * 1000)  # Convert from 0-1 to 0-1000
            gripper_value = np.clip(gripper_value, 0, 1000)
            
            # Safety check: reject if joint delta is too large
            if self._current_joint_angles is not None:
                joint_delta = np.abs(target_joints - self._current_joint_angles)
                max_delta = np.max(joint_delta)
                
                if max_delta > self.max_joint_delta_rad:
                    if self.debug:
                        logger.warning(
                            f"[AliciaD] Action rejected: max joint delta {np.rad2deg(max_delta):.1f}° "
                            f"exceeds limit {np.rad2deg(self.max_joint_delta_rad):.1f}°"
                        )
                    return  # Reject action
            
            # Send to hardware
            self.servo_driver.set_joint_and_gripper(
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
        return self._connected and self.servo_driver.serial_comm.is_connected()
    
    def shutdown(self):
        """Shutdown robot."""
        try:
            self.servo_driver.stop_update_thread()
            self.servo_driver.disconnect()
            self._connected = False
            logger.info("[AliciaD] Shutdown complete")
        except Exception as e:
            logger.error(f"[AliciaD] Shutdown error: {e}")
    
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
                images = images[None]  # Add batch dimension
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
        home_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Gripper open
        self.publish_action(home_action)
    
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
            self.ik_solver.reset(self._current_joint_angles)
        return self.ik_solver.get_current_pose()
    
    # ==================== Async Start (like SO101) ====================
    
    def start(self):
        """
        Start the robot with async architecture.
        
        Key design for imitation learning:
        - Observation is ALWAYS fresh (synchronous blocking read)
        - IK computation runs in separate process (never blocks obs)
        - Action execution is non-blocking
        """
        import multiprocessing as mp
        from deploy.utils import RateLimiter
        
        # Create shared memory for observations
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Connect to control shm (teleop)
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
        
        # Choose start mode based on control mode
        if self.control_mode == "delta_ee":
            self._start_async_ik()
        else:
            self._start_sync()
    
    def _start_sync(self):
        """Synchronous start (for qpos mode - no IK needed)."""
        from deploy.utils import RateLimiter
        
        self.is_running = True
        rate_limiter = RateLimiter()
        
        while self.is_running:
            # 1. Get fresh observation (blocking)
            data = self.get_data()
            if data is not None:
                self.write_data(data)
            
            # 2. Process and execute action
            action = self.read_action()
            if action is not None:
                action = self.process_action(action)
                action_array = action.get('action', None)
                if action_array is not None:
                    self.publish_action(action_array)
            
            rate_limiter.sleep(self.fps)
    
    def _start_async_ik(self):
        """
        Async start with IK in separate process.
        
        Architecture (like SO101):
        - Main loop: obs acquisition (blocking) + action execution (non-blocking)
        - IK process: delta_ee → joint angles conversion
        - Queues: decouple obs rate from IK rate
        """
        import multiprocessing as mp
        from deploy.utils import RateLimiter
        
        # Create queues for IK process communication
        action_in_queue = mp.Queue(maxsize=2)
        action_out_queue = mp.Queue(maxsize=2)
        
        # Start IK worker process
        ik_process = mp.Process(
            target=self._ik_worker,
            args=(action_in_queue, action_out_queue),
            daemon=True
        )
        ik_process.start()
        logger.info(f"[AliciaD] Started async IK process")
        
        self.is_running = True
        rate_limiter = RateLimiter()
        pending_result = None
        
        try:
            while self.is_running:
                # 1. Get FRESH observation (blocking, ~50-100ms)
                #    This is the ground truth for imitation learning
                data = self.get_data()
                if data is not None:
                    self.write_data(data)
                
                # 2. Check for processed IK result (non-blocking)
                try:
                    while not action_out_queue.empty():
                        pending_result = action_out_queue.get_nowait()
                except:
                    pass
                
                # 3. Execute pending action (non-blocking serial write)
                if pending_result is not None:
                    action_array = pending_result.get('action', None)
                    if action_array is not None:
                        self.publish_action(action_array)
                    
                    # Update IK state tracking
                    with self._lock:
                        if pending_result.get('target_qpos') is not None:
                            self._target_qpos = np.array(pending_result['target_qpos'])
                            self._target_gripper = pending_result['target_gripper']
                    
                    pending_result = None
                
                # 4. Send new raw action to IK process (non-blocking)
                raw_action = self.read_action()
                if raw_action is not None:
                    with self._lock:
                        msg = {
                            'action': raw_action,
                            'target_qpos': self._target_qpos.copy() if self._target_qpos is not None else None,
                            'target_gripper': self._target_gripper,
                        }
                    try:
                        # Drop old messages to avoid lag
                        while not action_in_queue.empty():
                            try:
                                action_in_queue.get_nowait()
                            except:
                                break
                        action_in_queue.put_nowait(msg)
                    except:
                        pass
                
                rate_limiter.sleep(self.fps)
        finally:
            # Cleanup
            action_in_queue.put(None)  # Signal to stop
            ik_process.join(timeout=1.0)
            if ik_process.is_alive():
                ik_process.terminate()
            logger.info(f"[AliciaD] Stopped async IK process")
    
    def _ik_worker(self, action_in_queue, action_out_queue):
        """
        IK worker process.
        
        Runs independently of main loop, converts delta_ee actions to joint angles.
        """
        # Create local IK solver (each process needs its own)
        local_solver = RealtimeIKSolver(
            urdf_path=self.urdf_path,
            end_frame="tool0",
            arm_dof=6,
            eps=self._ik_settings['eps'],
            max_iter=self._ik_settings['max_iter'],
            dt=self._ik_settings['dt'],
        )
        
        target_gripper = 500.0
        
        while True:
            try:
                msg = action_in_queue.get(timeout=1.0)
                if msg is None:
                    break
                
                raw_action = msg['action']
                action = raw_action.get('action', None)
                if action is None:
                    continue
                
                action = np.array(action, dtype=np.float64)
                if len(action) != 7:
                    continue
                
                # Initialize solver state from message
                if msg.get('target_qpos') is not None:
                    local_solver.reset(msg['target_qpos'])
                    target_gripper = msg['target_gripper']
                
                # Parse delta action
                # Input: [dx, dy, dz, droll, dpitch, dyaw, dgripper]
                # Teleop (Quest3) outputs xyz euler angles in BASE frame after calibration_transform
                delta_pos = action[:3] * self.position_scale
                euler_delta = action[3:6] * self.rotation_scale  # xyz euler angles in base frame
                delta_gripper = action[6] * self.gripper_scale
                
                # Skip if essentially zero
                if np.allclose(action, 0, atol=1e-6):
                    continue
                
                # Get current end-effector pose from solver's internal state
                current_T = local_solver.get_current_pose()
                
                # Build target pose
                target_T = current_T.copy()
                target_T[:3, 3] += delta_pos  # Apply position delta
                
                # Convert euler delta to rotation matrix (base frame delta rotation)
                # Then LEFT multiply to current rotation (rotation in base/world frame)
                if np.linalg.norm(euler_delta) > 1e-10:
                    from scipy.spatial.transform import Rotation as R
                    R_delta_base = R.from_euler('xyz', euler_delta).as_matrix()
                    # Left multiply: delta rotation is in base frame
                    # target_R = R_delta_base @ current_R
                    target_T[:3, :3] = R_delta_base @ current_T[:3, :3]
                
                # Solve IK with target pose
                success, q_arm, iters, err = local_solver.solve(target_T)
                
                # SAFETY: If IK failed, skip this action entirely
                # Do NOT update gripper or send result - target is unreachable
                if not success:
                    # Don't update target_gripper or solver state
                    continue
                
                # IK succeeded - update gripper and send result
                new_gripper = np.clip(target_gripper + delta_gripper * 1000, 0, 1000)
                target_gripper = new_gripper
                
                # Build output
                output_action = np.concatenate([q_arm, [new_gripper / 1000.0]])
                
                result = {
                    'action': output_action,
                    'target_qpos': q_arm.copy(),
                    'target_gripper': new_gripper,
                    'ik_success': success,
                    'ik_iters': iters,
                    'ik_err': err,
                }
                
                # Send result (drop old if queue full)
                try:
                    while not action_out_queue.empty():
                        try:
                            action_out_queue.get_nowait()
                        except:
                            break
                    action_out_queue.put_nowait(result)
                except:
                    pass
                    
            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    logger.error(f"[AliciaD IK Worker] Error: {e}")
                continue


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Alicia-D Robot")
    parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port')
    parser.add_argument('--mode', type=str, default='qpos', choices=['qpos', 'delta_ee'], help='Control mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    print(f"Creating Alicia-D robot on {args.port} with mode={args.mode}")
    
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
