#!/usr/bin/env python
# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# IK Benchmark: Compare Pinocchio vs RoboCore inverse kinematics solvers
#
# This script compares:
# - Computation time
# - Position/orientation accuracy
# - Solution correctness (verified via forward kinematics)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Real-time IK Solver for VR Teleoperation
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
        eps: float = 1e-4,
        max_iter: int = 50,
        dt: float = 1.0,
        damp: float = 1e-6,
    ):
        """
        Initialize real-time IK solver.
        
        Args:
            urdf_path: Path to URDF file
            end_frame: End-effector frame name
            arm_dof: Number of arm joints (excludes gripper)
            eps: Convergence tolerance
            max_iter: Maximum iterations per solve
            dt: Step size (larger = faster but may overshoot)
            damp: Damping factor for DLS
        """
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
        """Get current joint configuration."""
        return self._q_current.copy()
    
    def solve(self, target_pose: np.ndarray) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve IK for target pose using warm start.
        
        Args:
            target_pose: 4x4 transformation matrix
            
        Returns:
            Tuple of (success, q, iterations, error_norm)
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
                self._q_current = q
                return True, q.copy(), i + 1, err_norm
            
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
        
        # Even if not converged, update state for next iteration
        self._q_current = q
        return False, q.copy(), self.max_iter, err_norm
    
    def solve_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray) -> Tuple[bool, np.ndarray, int, float]:
        """
        Solve IK for incremental motion (common in VR teleoperation).
        
        Args:
            delta_pos: Position delta [dx, dy, dz] in meters
            delta_rot: Rotation delta as axis-angle [rx, ry, rz] in radians
            
        Returns:
            Tuple of (success, q, iterations, error_norm)
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
# Pinocchio IK Solver (for benchmarking)
# ============================================================================

def setup_pinocchio(urdf_path: str, end_frame: str = "tool0"):
    """Setup Pinocchio model and data."""
    import pinocchio as pin
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    try:
        frame_id = model.getFrameId(end_frame)
    except ValueError:
        raise ValueError(f"Frame '{end_frame}' not found in URDF. Available frames: {[f.name for f in model.frames]}")
    
    return model, data, frame_id


def pinocchio_fk(model, data, frame_id, q: np.ndarray) -> np.ndarray:
    """Forward kinematics using Pinocchio. Returns 4x4 transform matrix."""
    import pinocchio as pin
    
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    pose = data.oMf[frame_id]
    T = np.eye(4)
    T[:3, :3] = pose.rotation
    T[:3, 3] = pose.translation
    return T


def pinocchio_ik_single(
    model, data, frame_id,
    target: "pin.SE3",
    q_init: np.ndarray,
    eps: float,
    max_iter: int,
    dt: float,
    damp: float,
    arm_dof: int,
) -> Tuple[bool, np.ndarray, int, float, float, float]:
    """Single attempt of Pinocchio IK."""
    import pinocchio as pin
    
    q = q_init.copy()
    pos_err = float('inf')
    ori_err = float('inf')
    total_err = float('inf')
    
    for i in range(max_iter):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Compute error in local frame
        current_pose = data.oMf[frame_id]
        dMi = current_pose.actInv(target)
        err = pin.log(dMi).vector  # 6D error (linear, angular)
        
        # Separate position and orientation errors
        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])
        total_err = np.linalg.norm(err)
        
        # Check convergence
        if total_err < eps:
            return True, q.copy(), i + 1, pos_err, ori_err, total_err
        
        # Compute Jacobian in local frame
        J_full = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
        
        # Only use arm joints (first arm_dof columns)
        J = J_full[:, :arm_dof]
        
        # Damped Least Squares solution (only for arm joints)
        JtJ = J.T @ J + np.eye(arm_dof) * damp
        v_arm = np.linalg.solve(JtJ, J.T @ err)
        
        # Build full velocity vector (arm + zero for gripper)
        v = np.zeros(model.nv)
        v[:arm_dof] = v_arm
        
        # Update joint angles
        q = pin.integrate(model, q, v * dt)
        
        # Clamp to joint limits
        q = np.maximum(q, model.lowerPositionLimit)
        q = np.minimum(q, model.upperPositionLimit)
    
    return False, q.copy(), max_iter, pos_err, ori_err, total_err


def pinocchio_ik(
    model, data, frame_id,
    target_pose: np.ndarray,  # 4x4 transform matrix
    q_init: np.ndarray = None,
    eps: float = 1e-4,
    max_iter: int = 500,
    dt: float = 0.1,
    damp: float = 1e-6,
    arm_dof: int = 6,  # Only solve for arm joints, ignore gripper
    num_initial_guesses: int = 1,  # Number of random initial guesses to try
) -> Dict[str, Any]:
    """
    Pinocchio-based IK solver using Damped Least Squares.
    
    Args:
        arm_dof: Number of arm joints to solve for (default 6, ignores gripper joints)
        num_initial_guesses: Number of initial guesses to try (1 = only q_init)
    
    Returns:
        Dictionary with 'success', 'q', 'iters', 'pos_err', 'ori_err', 'computation_time'
    """
    import pinocchio as pin
    
    start_time = time.perf_counter()
    
    # Convert 4x4 matrix to SE3
    target = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
    
    q_default = q_init.copy() if q_init is not None else pin.neutral(model)
    
    best_result = None
    best_err = float('inf')
    total_iters = 0
    
    # Try with q_init first
    initial_guesses = [q_default]
    
    # Generate random initial guesses
    if num_initial_guesses > 1:
        for _ in range(num_initial_guesses - 1):
            # Random joint angles within 80% of limits
            q_random = np.random.uniform(
                model.lowerPositionLimit * 0.8,
                model.upperPositionLimit * 0.8
            )
            # Keep gripper joints at default
            q_random[arm_dof:] = q_default[arm_dof:]
            initial_guesses.append(q_random)
    
    for q_guess in initial_guesses:
        success, q, iters, pos_err, ori_err, err_norm = pinocchio_ik_single(
            model, data, frame_id, target, q_guess, eps, max_iter, dt, damp, arm_dof
        )
        total_iters += iters
        
        if success:
            elapsed = time.perf_counter() - start_time
            return {
                'success': True,
                'q': q,
                'iters': total_iters,
                'pos_err': pos_err,
                'ori_err': ori_err,
                'err_norm': err_norm,
                'computation_time': elapsed,
            }
        
        # Keep track of best result even if not converged
        if err_norm < best_err:
            best_err = err_norm
            best_result = (q, pos_err, ori_err, err_norm)
    
    elapsed = time.perf_counter() - start_time
    q, pos_err, ori_err, err_norm = best_result
    return {
        'success': False,
        'q': q,
        'iters': total_iters,
        'pos_err': pos_err,
        'ori_err': ori_err,
        'err_norm': err_norm,
        'computation_time': elapsed,
        'message': f'Did not converge after {num_initial_guesses} attempts',
    }


# ============================================================================
# RoboCore IK Solver
# ============================================================================

def setup_robocore(urdf_path: str, base_link: str = "base_link", end_link: str = "tool0"):
    """Setup RoboCore robot model."""
    from robocore.modeling import RobotModel
    
    robot_model = RobotModel(urdf_path, base_link=base_link, end_link=end_link)
    # Note: RoboCore IK uses chain DOF (num_chain_dof), not total model DOF
    return robot_model


def get_robocore_chain_dof(robot_model) -> int:
    """Get the DOF of the kinematic chain used for IK."""
    # The chain DOF is the number of joints in the kinematic chain
    return robot_model.num_chain_dof


def robocore_fk(robot_model, q: np.ndarray) -> np.ndarray:
    """Forward kinematics using RoboCore. Returns 4x4 transform matrix."""
    from robocore.kinematics import forward_kinematics
    from robocore.utils.backend import to_numpy
    
    # return_end=True returns only the end-effector pose as 4x4 matrix
    result = forward_kinematics(robot_model, q, return_end=True)
    if isinstance(result, dict):
        # Fallback if dict returned (shouldn't happen with return_end=True)
        return to_numpy(result[robot_model.end_link])
    return to_numpy(result) if not isinstance(result, np.ndarray) else result


def robocore_ik(
    robot_model,
    target_pose: np.ndarray,  # 4x4 transform matrix
    q_init: np.ndarray = None,
    method: str = 'dls',
    max_iters: int = 500,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-4,
    num_initial_guesses: int = 10,
    initial_guess_strategy: str = 'random',
) -> Dict[str, Any]:
    """
    RoboCore-based IK solver.
    
    Returns:
        Dictionary with 'success', 'q', 'iters', 'pos_err', 'ori_err', 'computation_time'
    """
    from robocore.kinematics import inverse_kinematics
    from robocore.utils.backend import to_numpy
    
    start_time = time.perf_counter()
    
    result = inverse_kinematics(
        robot_model,
        target_pose,
        q0=q_init,
        method=method,
        max_iters=max_iters,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        num_initial_guesses=num_initial_guesses,
        initial_guess_strategy=initial_guess_strategy,
        use_analytic_jacobian=True,
    )
    
    elapsed = time.perf_counter() - start_time
    
    # Extract and normalize results
    q = to_numpy(result['q']) if result.get('q') is not None else None
    success = result.get('success', False)
    if isinstance(success, list):
        success = success[0] if success else False
    
    iters = result.get('iters', 0)
    if isinstance(iters, list):
        iters = iters[0] if iters else 0
    
    pos_err = result.get('pos_err', float('inf'))
    if isinstance(pos_err, list):
        pos_err = pos_err[0] if pos_err else float('inf')
    
    ori_err = result.get('ori_err', float('inf'))
    if isinstance(ori_err, list):
        ori_err = ori_err[0] if ori_err else float('inf')
    
    err_norm = result.get('err_norm', None)
    if isinstance(err_norm, list):
        err_norm = err_norm[0] if err_norm else None
    
    return {
        'success': success,
        'q': q,
        'iters': iters,
        'pos_err': pos_err,
        'ori_err': ori_err,
        'err_norm': err_norm,
        'computation_time': elapsed,
    }


# ============================================================================
# Benchmark Utilities
# ============================================================================

def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    
    # Normalize
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    """Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transform matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = quaternion_to_matrix(pose[3:7])
    return T


def compute_pose_error(T1: np.ndarray, T2: np.ndarray) -> Tuple[float, float]:
    """Compute position and orientation error between two 4x4 transforms."""
    # Position error
    pos_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    
    # Orientation error (using rotation matrix difference)
    R_diff = T1[:3, :3].T @ T2[:3, :3]
    trace = np.trace(R_diff)
    # Clamp trace to [-1, 3] for numerical stability
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)
    ori_err = abs(angle)
    
    return pos_err, ori_err


def generate_test_poses(model, data, frame_id, num_poses: int = 5) -> List[np.ndarray]:
    """Generate test poses by random joint configurations and FK."""
    import pinocchio as pin
    
    poses = []
    
    # Start with neutral pose
    q_neutral = pin.neutral(model)
    pin.forwardKinematics(model, data, q_neutral)
    pin.updateFramePlacements(model, data)
    T = np.eye(4)
    T[:3, :3] = data.oMf[frame_id].rotation
    T[:3, 3] = data.oMf[frame_id].translation
    poses.append(T.copy())
    
    # Generate random poses within joint limits
    np.random.seed(42)  # For reproducibility
    for _ in range(num_poses - 1):
        q_random = np.random.uniform(
            model.lowerPositionLimit * 0.8,  # Use 80% of limits for safety
            model.upperPositionLimit * 0.8
        )
        pin.forwardKinematics(model, data, q_random)
        pin.updateFramePlacements(model, data)
        T = np.eye(4)
        T[:3, :3] = data.oMf[frame_id].rotation
        T[:3, 3] = data.oMf[frame_id].translation
        poses.append(T.copy())
    
    return poses


@dataclass
class BenchmarkResult:
    name: str
    success: bool
    time_ms: float
    pos_err: float
    ori_err: float
    iters: int
    q: np.ndarray
    verified_pos_err: float = None
    verified_ori_err: float = None


def print_comparison_table(results: List[Tuple[BenchmarkResult, BenchmarkResult]], test_names: List[str]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("IK BENCHMARK RESULTS")
    print("=" * 100)
    
    headers = ["Test", "Solver", "Success", "Time(ms)", "Pos Err(m)", "Ori Err(rad)", "Iters", "FK Verify Pos", "FK Verify Ori"]
    col_widths = [12, 12, 8, 10, 12, 12, 8, 14, 14]
    
    # Print header
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    for i, ((pin_res, rc_res), test_name) in enumerate(zip(results, test_names)):
        # Pinocchio row
        pin_row = [
            test_name[:col_widths[0]],
            "Pinocchio",
            "✓" if pin_res.success else "✗",
            f"{pin_res.time_ms:.2f}",
            f"{pin_res.pos_err:.2e}" if pin_res.pos_err < float('inf') else "N/A",
            f"{pin_res.ori_err:.2e}" if pin_res.ori_err < float('inf') else "N/A",
            str(pin_res.iters),
            f"{pin_res.verified_pos_err:.2e}" if pin_res.verified_pos_err is not None else "N/A",
            f"{pin_res.verified_ori_err:.2e}" if pin_res.verified_ori_err is not None else "N/A",
        ]
        print(" | ".join(str(v).center(w) for v, w in zip(pin_row, col_widths)))
        
        # RoboCore row
        rc_row = [
            "",
            "RoboCore",
            "✓" if rc_res.success else "✗",
            f"{rc_res.time_ms:.2f}",
            f"{rc_res.pos_err:.2e}" if rc_res.pos_err < float('inf') else "N/A",
            f"{rc_res.ori_err:.2e}" if rc_res.ori_err < float('inf') else "N/A",
            str(rc_res.iters),
            f"{rc_res.verified_pos_err:.2e}" if rc_res.verified_pos_err is not None else "N/A",
            f"{rc_res.verified_ori_err:.2e}" if rc_res.verified_ori_err is not None else "N/A",
        ]
        print(" | ".join(str(v).center(w) for v, w in zip(rc_row, col_widths)))
        print("-" * len(header_line))
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    pin_times = [r[0].time_ms for r in results if r[0].success]
    rc_times = [r[1].time_ms for r in results if r[1].success]
    
    pin_success = sum(1 for r in results if r[0].success)
    rc_success = sum(1 for r in results if r[1].success)
    
    print(f"\nPinocchio:")
    print(f"  Success rate: {pin_success}/{len(results)} ({100*pin_success/len(results):.1f}%)")
    if pin_times:
        print(f"  Avg time: {np.mean(pin_times):.2f} ms")
        print(f"  Min time: {np.min(pin_times):.2f} ms")
        print(f"  Max time: {np.max(pin_times):.2f} ms")
    
    print(f"\nRoboCore:")
    print(f"  Success rate: {rc_success}/{len(results)} ({100*rc_success/len(results):.1f}%)")
    if rc_times:
        print(f"  Avg time: {np.mean(rc_times):.2f} ms")
        print(f"  Min time: {np.min(rc_times):.2f} ms")
        print(f"  Max time: {np.max(rc_times):.2f} ms")
    
    if pin_times and rc_times:
        speedup = np.mean(rc_times) / np.mean(pin_times)
        print(f"\nSpeedup (RoboCore avg / Pinocchio avg): {speedup:.2f}x")
        if speedup > 1:
            print(f"  → Pinocchio is {speedup:.2f}x faster on average")
        else:
            print(f"  → RoboCore is {1/speedup:.2f}x faster on average")


def main(args):
    print("=" * 60)
    print("IK Benchmark: Pinocchio vs RoboCore")
    print("=" * 60)
    
    urdf_path = args.urdf
    print(f"\nURDF: {urdf_path}")
    
    # Setup solvers
    print("\nSetting up solvers...")
    
    # Pinocchio
    pin_model, pin_data, pin_frame_id = setup_pinocchio(urdf_path, args.end_link)
    print(f"  Pinocchio: DoF={pin_model.nv}, Frame='{args.end_link}' (id={pin_frame_id})")
    
    # RoboCore
    rc_model = setup_robocore(urdf_path, args.base_link, args.end_link)
    rc_chain_dof = rc_model.num_chain_dof
    print(f"  RoboCore: Total DoF={rc_model.num_dof}, Chain DoF={rc_chain_dof}")
    
    # Generate or use specified test poses
    if args.end_pose:
        # Use user-specified pose
        test_poses = [pose_to_matrix(args.end_pose)]
        test_names = ["User Pose"]
    else:
        # Generate test poses
        print(f"\nGenerating {args.num_tests} test poses...")
        test_poses = generate_test_poses(pin_model, pin_data, pin_frame_id, args.num_tests)
        test_names = [f"Test {i+1}" for i in range(len(test_poses))]
    
    # Run benchmark
    print(f"\nRunning benchmark with {args.num_runs} runs per test...")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Tolerance: pos={args.pos_tol}, ori={args.ori_tol}")
    
    import pinocchio as pin
    q_init_pin = pin.neutral(pin_model)
    # For RoboCore, only use arm joints (exclude gripper joints)
    q_init_rc = q_init_pin[:rc_chain_dof].copy()
    
    print(f"\n  Initial q (Pinocchio, {len(q_init_pin)} joints): {np.rad2deg(q_init_pin)}")
    print(f"  Initial q (RoboCore chain, {len(q_init_rc)} joints): {np.rad2deg(q_init_rc)}")
    
    all_results = []
    
    for test_idx, (target_pose, test_name) in enumerate(zip(test_poses, test_names)):
        print(f"\n--- {test_name} ---")
        print(f"Target position: [{target_pose[0,3]:.4f}, {target_pose[1,3]:.4f}, {target_pose[2,3]:.4f}]")
        
        # Run multiple times and take average
        pin_times = []
        rc_times = []
        pin_result = None
        rc_result = None
        
        for run in range(args.num_runs):
            # Pinocchio IK
            pin_res = pinocchio_ik(
                pin_model, pin_data, pin_frame_id,
                target_pose,
                q_init=q_init_pin,
                eps=args.pos_tol,
                max_iter=args.max_iters,
                dt=args.dt,
                damp=args.damp,
                arm_dof=args.arm_dof,
                num_initial_guesses=args.pin_num_inits,
            )
            pin_times.append(pin_res['computation_time'] * 1000)
            if run == 0:
                pin_result = pin_res
            
            # RoboCore IK (only use chain DOF for q_init)
            rc_res = robocore_ik(
                rc_model,
                target_pose,
                q_init=q_init_rc,
                method=args.method,
                max_iters=args.max_iters,
                pos_tol=args.pos_tol,
                ori_tol=args.ori_tol,
                num_initial_guesses=args.num_inits,
                initial_guess_strategy=args.init_strategy,
            )
            rc_times.append(rc_res['computation_time'] * 1000)
            if run == 0:
                rc_result = rc_res
        
        # Verify results using FK
        pin_verified_pos, pin_verified_ori = None, None
        rc_verified_pos, rc_verified_ori = None, None
        
        if pin_result['q'] is not None:
            pin_fk_pose = pinocchio_fk(pin_model, pin_data, pin_frame_id, pin_result['q'])
            pin_verified_pos, pin_verified_ori = compute_pose_error(target_pose, pin_fk_pose)
        
        if rc_result['q'] is not None:
            rc_fk_pose = robocore_fk(rc_model, rc_result['q'])
            rc_verified_pos, rc_verified_ori = compute_pose_error(target_pose, rc_fk_pose)
        
        # Create result objects
        pin_bench = BenchmarkResult(
            name="Pinocchio",
            success=pin_result['success'],
            time_ms=np.mean(pin_times),
            pos_err=pin_result['pos_err'],
            ori_err=pin_result['ori_err'],
            iters=pin_result['iters'],
            q=pin_result['q'],
            verified_pos_err=pin_verified_pos,
            verified_ori_err=pin_verified_ori,
        )
        
        rc_bench = BenchmarkResult(
            name="RoboCore",
            success=rc_result['success'],
            time_ms=np.mean(rc_times),
            pos_err=rc_result['pos_err'],
            ori_err=rc_result['ori_err'],
            iters=rc_result['iters'],
            q=rc_result['q'],
            verified_pos_err=rc_verified_pos,
            verified_ori_err=rc_verified_ori,
        )
        
        all_results.append((pin_bench, rc_bench))
        
        # Quick print
        print(f"  Pinocchio: {'✓' if pin_bench.success else '✗'} {pin_bench.time_ms:.2f}ms, "
              f"pos_err={pin_bench.pos_err:.2e}, ori_err={pin_bench.ori_err:.2e}")
        print(f"  RoboCore:  {'✓' if rc_bench.success else '✗'} {rc_bench.time_ms:.2f}ms, "
              f"pos_err={rc_bench.pos_err:.2e}, ori_err={rc_bench.ori_err:.2e}")
        
        if args.verbose:
            print(f"\n  Pinocchio q (deg): {np.rad2deg(pin_result['q'])}")
            print(f"  RoboCore  q (deg): {np.rad2deg(rc_result['q']) if rc_result['q'] is not None else 'N/A'}")
    
    # Print comparison table
    print_comparison_table(all_results, test_names)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


def benchmark_realtime(args):
    """Benchmark real-time IK solver for VR teleoperation scenario."""
    print("=" * 60)
    print("Real-time IK Benchmark (VR Teleoperation Scenario)")
    print("=" * 60)
    
    # Initialize solver
    solver = RealtimeIKSolver(
        urdf_path=args.urdf,
        end_frame=args.end_link,
        arm_dof=args.arm_dof,
        eps=args.pos_tol,
        max_iter=args.rt_max_iters,
        dt=args.rt_dt,
        damp=args.damp,
    )
    
    print(f"\nURDF: {args.urdf}")
    print(f"Solver params: eps={args.pos_tol}, max_iter={args.rt_max_iters}, dt={args.rt_dt}")
    
    # Simulate continuous VR controller motion
    np.random.seed(42)
    num_steps = args.rt_num_steps
    
    print(f"\nSimulating {num_steps} steps of continuous motion...")
    print(f"  Position delta: ±{args.rt_pos_delta*1000:.1f}mm per step")
    print(f"  Rotation delta: ±{np.rad2deg(args.rt_rot_delta):.1f}° per step")
    
    times = []
    successes = 0
    errors = []
    iterations = []
    
    for step in range(num_steps):
        # Random small delta (simulating VR controller movement)
        delta_pos = np.random.randn(3) * args.rt_pos_delta
        delta_rot = np.random.randn(3) * args.rt_rot_delta
        
        start = time.perf_counter()
        success, q, iters, err = solver.solve_delta(delta_pos, delta_rot)
        elapsed = (time.perf_counter() - start) * 1000
        
        times.append(elapsed)
        iterations.append(iters)
        errors.append(err)
        if success:
            successes += 1
    
    times = np.array(times)
    iterations = np.array(iterations)
    errors = np.array(errors)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nSuccess rate: {successes}/{num_steps} ({100*successes/num_steps:.1f}%)")
    
    print(f"\nTiming Statistics:")
    print(f"  Average:         {times.mean():.3f} ms")
    print(f"  Std deviation:   {times.std():.3f} ms")
    print(f"  Minimum:         {times.min():.3f} ms")
    print(f"  Maximum:         {times.max():.3f} ms")
    print(f"  Median:          {np.median(times):.3f} ms")
    print(f"  90th percentile: {np.percentile(times, 90):.3f} ms")
    print(f"  95th percentile: {np.percentile(times, 95):.3f} ms")
    print(f"  99th percentile: {np.percentile(times, 99):.3f} ms")
    
    print(f"\nIterations Statistics:")
    print(f"  Average: {iterations.mean():.1f}")
    print(f"  Max:     {iterations.max()}")
    
    print(f"\nError Statistics (when converged):")
    converged_errors = errors[errors < args.pos_tol * 10]  # Filter out non-converged
    if len(converged_errors) > 0:
        print(f"  Average error: {converged_errors.mean():.2e}")
        print(f"  Max error:     {converged_errors.max():.2e}")
    
    # Calculate achievable control frequency
    avg_time = times.mean()
    max_freq = 1000 / avg_time
    safe_freq = 1000 / np.percentile(times, 99)
    
    print(f"\nAchievable Control Frequency:")
    print(f"  Based on average:         {max_freq:.0f} Hz")
    print(f"  Based on 99th percentile: {safe_freq:.0f} Hz (recommended)")
    
    # VR teleoperation requirements
    print(f"\nVR Teleoperation Compatibility:")
    requirements = [
        ("Quest 3 (72Hz)", 72, 1000/72),
        ("Quest 3 (90Hz)", 90, 1000/90),
        ("Quest 3 (120Hz)", 120, 1000/120),
        ("High-frequency control (500Hz)", 500, 2.0),
        ("Real-time servo (1000Hz)", 1000, 1.0),
    ]
    
    for name, freq, max_time in requirements:
        p99 = np.percentile(times, 99)
        status = "✓" if p99 < max_time else "✗"
        margin = max_time - p99
        print(f"  {status} {name}: need <{max_time:.2f}ms, got {p99:.3f}ms (margin: {margin:+.2f}ms)")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IK Benchmark: Pinocchio vs RoboCore")
    
    # Mode selection
    parser.add_argument('--realtime', '-rt', action='store_true',
        help="Run real-time IK benchmark for VR teleoperation")
    
    # URDF settings
    parser.add_argument('--urdf', type=str, 
        default="/media/zzz/ElementsSE/laptop_ubuntu/Codes/ILStudio/.venv/lib/python3.10/site-packages/synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf",
        help="Path to URDF file")
    parser.add_argument('--base-link', type=str, default="base_link", help="Base link name")
    parser.add_argument('--end-link', type=str, default="tool0", help="End effector link name")
    
    # Test settings
    parser.add_argument('--num-tests', type=int, default=5, help="Number of random test poses")
    parser.add_argument('--num-runs', type=int, default=3, help="Number of runs per test (for timing)")
    parser.add_argument('--end-pose', type=float, nargs=7, default=None,
        help='Specific target pose [x y z qx qy qz qw] (overrides random tests)')
    
    # IK settings
    parser.add_argument('--max-iters', type=int, default=500, help="Maximum iterations")
    parser.add_argument('--pos-tol', type=float, default=1e-4, help="Position tolerance (m)")
    parser.add_argument('--ori-tol', type=float, default=1e-4, help="Orientation tolerance (rad)")
    
    # Pinocchio-specific
    parser.add_argument('--dt', type=float, default=0.1, help="Pinocchio step size")
    parser.add_argument('--damp', type=float, default=1e-6, help="Pinocchio damping factor")
    parser.add_argument('--arm-dof', type=int, default=6, help="Number of arm DOF for Pinocchio IK (excludes gripper)")
    parser.add_argument('--pin-num-inits', type=int, default=1, help="Number of initial guesses for Pinocchio IK")
    
    # Real-time IK settings
    parser.add_argument('--rt-max-iters', type=int, default=50, help="Max iterations for real-time IK")
    parser.add_argument('--rt-dt', type=float, default=1.0, help="Step size for real-time IK")
    parser.add_argument('--rt-num-steps', type=int, default=1000, help="Number of steps for real-time benchmark")
    parser.add_argument('--rt-pos-delta', type=float, default=0.005, help="Position delta per step (m)")
    parser.add_argument('--rt-rot-delta', type=float, default=0.05, help="Rotation delta per step (rad)")
    
    # RoboCore-specific
    parser.add_argument('--method', type=str, default='dls', choices=['dls', 'pinv', 'transpose'],
        help="RoboCore IK method")
    parser.add_argument('--num-inits', type=int, default=10, help="Number of initial guesses for RoboCore")
    parser.add_argument('--init-strategy', type=str, default='random',
        choices=['zero', 'random', 'sobol', 'latin', 'center', 'uniform'],
        help="Initial guess strategy for RoboCore")
    
    # Output settings
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    if args.realtime:
        benchmark_realtime(args)
    else:
        main(args)
