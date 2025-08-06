import numpy as np
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.linalg import norm, det

# ---------- 工具 ----------
def path_length(arr):
    return np.sum(norm(np.diff(arr, axis=0), axis=1))

def jerk(arr, dt):
    cs = CubicSpline(np.arange(arr.shape[0]), arr, axis=0)
    jerk_vec = cs(np.arange(2, arr.shape[0]-1), 3)   # 3 阶导
    return np.sum(norm(jerk_vec, axis=1)**2) * dt**5

def curvature_integral(arr):
    # 用三次样条求导
    t = np.arange(arr.shape[0])
    cs = CubicSpline(t, arr, axis=0)
    vel  = cs(t, 1)
    acc  = cs(t, 2)
    cross = np.cross(vel[:-1], acc[:-1])
    kappa = norm(cross, axis=1) / (norm(vel[:-1], axis=1)**3 + 1e-9)
    return np.sum(kappa) / arr.shape[0]

def effort(joint, dt):
    # 简化：effort ≈ Σ ||Δq||² / dt²
    return np.sum(norm(np.diff(joint, axis=0), axis=1)**2) / dt**2

def manipulability(joint):
    # 简化的 6×7 雅可比（仅演示用，真实机器人需调用 FK）
    # 这里用数值雅可比做占位
    J = np.random.randn(6, 7)  # 实际请替换为真实雅可比
    w = np.sqrt(det(J @ J.T + 1e-6 * np.eye(6)))
    return w

def dist_to_limits(joint, q_lb, q_ub):
    return np.min(np.concatenate([joint - q_lb, q_ub - joint]))

def legibility(xyz, goal_xyz):
    # 简化版 Dragan legibility: 平均方向与目标方向的余弦
    dirs = xyz[1:] - xyz[:-1]
    goal_dir = goal_xyz - xyz[:-1]
    cos = np.sum(dirs * goal_dir, axis=1) / (norm(dirs, axis=1) * norm(goal_dir, axis=1) + 1e-9)
    return np.mean(np.maximum(0, cos))

# ---------- 主函数 ----------
def compute_consistency_metrics(xyz, joint, dt, q_lb, q_ub, goal_xyz=None):
    metrics = {}
    # 笛卡尔空间
    metrics['path_length_cart'] = path_length(xyz)
    metrics['jerk_cart']        = jerk(xyz, dt)
    metrics['curvature_cart']   = curvature_integral(xyz)
    metrics['manipulability']   = manipulability(joint)  # 占位
    metrics['legibility']       = legibility(xyz, goal_xyz) if goal_xyz is not None else 0.0

    # 关节空间
    metrics['path_length_joint'] = path_length(joint)
    metrics['jerk_joint']        = jerk(joint, dt)
    metrics['curvature_joint']   = curvature_integral(joint)
    metrics['effort']            = effort(joint, dt)
    metrics['dist_to_limits']    = dist_to_limits(joint, q_lb, q_ub)

    return metrics

class ConsistencyMattersFilter:
    def __init__(self):
        pass
    
    def extract_feat(self, episode):
        return episode

    def __call__(self, dataset):
        return
        
    
# ---------- 使用示例 ----------
if __name__ == "__main__":
    N = 300
    dt = 0.033  # 30 Hz
    xyz   = np.random.randn(N, 3) * 0.2
    joint = np.random.randn(N, 7) * 0.1
    q_lb = -np.pi * np.ones(7)
    q_ub =  np.pi * np.ones(7)
    goal_xyz = np.array([0.2, 0.1, 0.05])

    m = compute_consistency_metrics(xyz, joint, dt, q_lb, q_ub, goal_xyz)
    for k, v in m.items():
        print(f"{k:25s}: {v:.4f}")