import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from deploy.teleoperator.base import BaseTeleopDevice, generate_shm_info
import rerun as rr
import time
from multiprocessing import shared_memory
import sys

# --- 跨平台输入检测 ---
if sys.platform == "win32":
    import msvcrt
else:
    import select


class IMUIntegrator:
    """
    只处理线性运动：接收世界坐标系下的加速度，并积分计算速度和位置。
    - acc_unit : m/s² (线性加速度)
    """

    def __init__(self, calib_samples: int = 200):
        self.N = calib_samples
        self.ACC_THRESHOLD = 0.5  # m/s^2
        self.reset()

    def reset(self):
        """重置所有状态和校准数据"""
        self.cnt = 0
        self.acc_bias = np.zeros(3)
        self.calibrated = False
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self._acc_samples = []
        print("\nIntegrator has been reset. Please keep phone stationary for recalibration.")

    def calibrate_step(self, acc_raw):
        """在校准阶段累积加速度数据"""
        if self.cnt < self.N:
            self._acc_samples.append(acc_raw)
            self.cnt += 1
            if self.cnt >= self.N:
                self._finalize_calibration()
        return self.calibrated

    def _finalize_calibration(self):
        """计算加速度计的零偏"""
        self.acc_bias = np.mean(self._acc_samples, axis=0)
        self.calibrated = True
        print("IMU 加速度计零偏校准完成。")
        print(f"  - 线性加速度零偏 (Lin Acc bias): {self.acc_bias}")

    def integrate_step(self, world_linear_acc, dt):
        """接收世界坐标系下的加速度，并更新速度和位置"""
        # 1. 始终进行速度积分
        self.velocity += world_linear_acc * dt

        # 2. 应用分轴独立连续动态阻尼
        gliding_friction = 0.998
        stopping_friction = 0.85

        for i in range(3):  # 分别处理 X, Y, Z 轴
            stopping_influence = np.clip(1.0 - abs(world_linear_acc[i]) / self.ACC_THRESHOLD, 0, 1)
            damping_factor = (1.0 - stopping_influence) * gliding_friction + stopping_influence * stopping_friction
            self.velocity[i] *= damping_factor

        # 3. 最终钳制
        if np.linalg.norm(self.velocity) < 0.01:
            self.velocity[:] = 0

        # 4. 更新位置
        self.position += self.velocity * dt


class IPhonePhyphox(BaseTeleopDevice):
    def __init__(self,
                 shm_name, shm_shape, shm_dtype,
                 action_dim=6,
                 action_dtype=np.float64,
                 frequency=30,
                 phyphox_ip="192.168.1.5",
                 phyphox_port=80,
                 calib_samples=150,
                 dt=0.1,
                 ):
        super().__init__(shm_name, shm_shape, shm_dtype,
                         action_dim=action_dim,
                         action_dtype=action_dtype,
                         frequency=frequency)

        if dt is not None:
            self.dt = dt
        else:
            self.dt = 1.0 / frequency

        # --- 核心修正: 请求线性加速度和欧拉角 ---
        # !!!重要!!! 请确保你的Phyphox实验将欧拉角输出命名为 yaw, pitch, roll
        self.url = f"http://{phyphox_ip}:{phyphox_port}/get?lin_accX&lin_accY&lin_accZ&yaw&pitch&roll"
        self.integrator = IMUIntegrator(calib_samples=calib_samples)

        # 用于存储姿态信息
        self.orientation = R.identity()
        self.initial_orientation_offset = None

    def get_observation(self):
        """获取线加速度和欧拉角 (度)"""
        try:
            response = requests.get(self.url, timeout=0.5)
            response.raise_for_status()
            data = response.json()["buffer"]
            ax = data["lin_accX"]["buffer"][-1]
            ay = data["lin_accY"]["buffer"][-1]
            az = data["lin_accZ"]["buffer"][-1]
            yaw = data["yaw"]["buffer"][-1]
            pitch = data["pitch"]["buffer"][-1]
            roll = data["roll"]["buffer"][-1]

            if any(v is None for v in [ax, ay, az, yaw, pitch, roll]):
                return None
            return np.array([ax, ay, az, yaw, pitch, roll])
        except Exception as e:
            print(f"Phyphox 读取错误: {e}")
            return None

    def get_absolute_pose(self):
        """从积分器获取绝对位姿"""
        pos = self.integrator.position
        abs_rot_euler = self.orientation.as_euler('xyz', degrees=False)
        return np.concatenate([pos, abs_rot_euler]).astype(self.action_dtype)

    def observation_to_action(self, obs):
        """满足抽象方法的要求"""
        return self.get_absolute_pose()

    def run(self):
        """主循环"""
        from deploy.robot.base import RateLimiter
        self.connect_to_buffer()
        rr.init("iphone_imu_trajectory_direct", spawn=True)
        rr.log("/", rr.ViewCoordinates.RFU, timeless=True)
        rr.log("world/origin_axes", rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=np.eye(3) * 0.1,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ), timeless=True)

        rate_limiter = RateLimiter()
        step = 0
        trajectory_points = []

        try:
            print("--- 直接使用轴角数据模型 ---")
            print("校准步骤: 将手机屏幕朝上平放，摄像头朝前，保持静止...")
            while not self.stop_event.is_set():
                reset_triggered = False
                if sys.platform == "win32":
                    if msvcrt.kbhit() and msvcrt.getch().decode('utf-8').lower() == 'r':
                        reset_triggered = True
                else:
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        if 'r' in line:
                            reset_triggered = True

                if reset_triggered:
                    self.integrator.reset()
                    self.initial_orientation_offset = None  # 重置姿态偏移
                    trajectory_points = []
                    rr.log("trajectory", rr.LineStrips3D([]))

                observation = self.get_observation()
                if observation is None:
                    rate_limiter.sleep(self.frequency)
                    continue

                acc_raw = observation[:3]
                yaw, pitch, roll = observation[3:]

                if not self.integrator.calibrated:
                    print(f"校准中... {self.integrator.cnt}/{self.integrator.N}", end='\r')
                    self.integrator.calibrate_step(acc_raw)
                    self.put_action_to_buffer(self.get_zero_action())
                else:
                    # --- 核心姿态处理逻辑 ---
                    # 1. 在校准后的第一帧，捕获初始姿态作为偏移量
                    if self.initial_orientation_offset is None:
                        # 注意: 欧拉角顺序 ZYX 对应 yaw, pitch, roll
                        self.initial_orientation_offset = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
                        print(f"捕获到初始姿态 (YPR degrees): {yaw:.1f}, {pitch:.1f}, {roll:.1f}")

                    # 2. 获取当前绝对姿态
                    current_absolute_rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)

                    # 3. 计算相对于初始姿态的旋转
                    self.orientation = current_absolute_rot * self.initial_orientation_offset.inv()

                    # 4. 使用更新后的姿态处理线性运动
                    acc_corr = acc_raw - self.integrator.acc_bias
                    world_linear_acc = self.orientation.apply(acc_corr)
                    self.integrator.integrate_step(world_linear_acc, self.dt)

                    # 5. 输出和可视化
                    absolute_pose = self.get_absolute_pose()
                    self.put_action_to_buffer(absolute_pose)

                    step += 1
                    rr.set_time_sequence("step", step)
                    pos_viz = self.integrator.position
                    orient_viz = self.orientation.as_quat()

                    rr.log("iphone", rr.Transform3D(translation=pos_viz, rotation=rr.Quaternion(xyzw=orient_viz)))
                    rr.log("iphone", rr.Boxes3D(half_sizes=[0.075, 0.035, 0.005]))

                    trajectory_points.append(pos_viz)
                    if len(trajectory_points) > 1:
                        rr.log("trajectory", rr.LineStrips3D([trajectory_points]))

                rate_limiter.sleep(self.frequency)
        finally:
            print("\n遥操作设备: 关闭中...")
            if self.shm: self.shm.close()


if __name__ == "__main__":
    from deploy.robot.base import RateLimiter

    action_dim = 6
    action_dtype = np.float64
    shm_info = generate_shm_info(shm_name='tmp_iphone_teleop', action_dim=action_dim, action_dtype=action_dtype)
    try:
        shm = shared_memory.SharedMemory(name=shm_info['name'], create=True, size=shm_info['size'])
        print(f"主程序: 已创建共享内存 '{shm_info['name']}'")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=shm_info['name'])
        print(f"主程序: 已连接到共享内存 '{shm_info['name']}'")
    try:
        IPHONE_IP = "172.20.10.1"
        dev = IPhonePhyphox(
            shm_name=shm_info['name'], shm_shape=shm_info['shape'], shm_dtype=shm_info['dtype'],
            action_dim=action_dim, action_dtype=action_dtype,
            frequency=30,
            phyphox_ip=IPHONE_IP, phyphox_port=80,
            calib_samples=90  # 30Hz下校准3秒
        )
        dev.run()
    finally:
        print("主程序: 清理共享内存。")
        shm.close()
        shm.unlink()
