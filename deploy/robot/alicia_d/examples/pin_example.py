import pinocchio as pin
import os
import numpy as np
import time

# 1. 指定 URDF 路径
urdf_path = "/media/zzz/ElementsSE/laptop_ubuntu/Codes/ILStudio/.venv/lib/python3.10/site-packages/synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"

# 2. 构建模型
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

print(f"机器人模型加载成功！DoF: {model.nv}")

# 3. 准备初始姿态
q = pin.neutral(model) 
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# --- 关键修改 1: 获取正确的末端 ID ---
try:
    FRAME_NAME = "tool0" # 或者是 "link6"，根据你之前的打印列表
    frame_id = model.getFrameId(FRAME_NAME)
except ValueError:
    print(f"Error: 找不到 Frame {FRAME_NAME}")
    exit()

# 获取当前末端位姿作为基准
start_pose = data.oMf[frame_id]
print(f"初始末端位置: {start_pose.translation.T}")

# --- 关键修改 2: 设定一个“简单且可达”的目标 ---
# 仅在当前位置基础上，Z轴向上移动 5cm (0.05m)
# 保持当前旋转不变，这样最容易求解成功
target_translation = start_pose.translation + np.array([0.0, 0.0, 0.05])
target_rotation = start_pose.rotation 
target = pin.SE3(target_rotation, target_translation)

print(f"目标末端位置: {target.translation.T}")

def solve_ik(model, data, target_pose, frame_id, q_init=None, eps=1e-4, max_iter=2000, dt=0.05, damp=1e-4):
    """
    修正后的 IK 求解器：增加了关节限位保护
    """
    q = q_init.copy() if q_init is not None else pin.neutral(model)
    
    for i in range(max_iter):
        # A. 正向运动学
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # B. 计算误差
        current_pose = data.oMf[frame_id]
        dMi = current_pose.actInv(target_pose)
        err = pin.log(dMi).vector 
        
        # 检查收敛
        if np.linalg.norm(err) < eps:
            print(f"IK 成功收敛！迭代次数: {i+1}, 最终误差: {np.linalg.norm(err):.6f}")
            return q, True
            
        # C. 计算雅可比
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
        
        # D. 求解 (Damped Least Squares)
        hessian = J.T @ J
        hessian += np.eye(model.nv) * damp
        gradient = J.T @ err
        v = np.linalg.solve(hessian, gradient)
        
        # E. 更新关节
        q = pin.integrate(model, q, v * dt)
        
        # --- 关键修改 3: 强制关节限位 (Clamping) ---
        # 这能防止出现 -67 这种离谱的数值
        q = np.maximum(q, model.lowerPositionLimit)
        q = np.minimum(q, model.upperPositionLimit)
        
    print(f"IK 未收敛。最终误差: {np.linalg.norm(err):.4f}")
    return q, False

# --- 运行求解 ---
t = time.time()
q_sol, success = solve_ik(model, data, target, frame_id, q_init=q)

print(f"耗时: {time.time() - t:.4f} 秒")
print(f"求解成功: {success}")
print(f"求解结果 q: {q_sol}")

# 验证一下结果
if success:
    pin.forwardKinematics(model, data, q_sol)
    pin.updateFramePlacements(model, data)
    final_pos = data.oMf[frame_id].translation
    print(f"验证-实际到达位置: {final_pos.T}")
    print(f"验证-位置误差: {np.linalg.norm(final_pos - target.translation):.6f}")