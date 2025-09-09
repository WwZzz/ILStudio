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


class IMUProcessor:
    """
    处理IMU数据，计算每一帧的相对位姿变换。
    不再进行状态累积，只计算瞬时变化量。
    - acc_unit : m/s² (线性加速度)
    - gyro_unit: rad/s
    """

    def __init__(self, calib_samples: int = 150):
        self.N = calib_samples
        self.ACC_THRESHOLD = 0.25  # m/s^2
        self.GYRO_THRESHOLD = 0.25  # rad/s
        self.reset()

    def reset(self):
        """重置校准数据"""
        self.cnt = 0
        self.lin_acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.calibrated = False
        self._lin_acc_samples = []
        self._gyro_samples = []
        print("\nIMU Processor has been reset. Please keep phone stationary for recalibration.")

    def calibrate_step(self, lin_acc_raw, gyro_raw):
        """在校准阶段累积数据"""
        if self.cnt < self.N:
            self._lin_acc_samples.append(lin_acc_raw)
            self._gyro_samples.append(gyro_raw)
            self.cnt += 1
            if self.cnt >= self.N:
                self._finalize_calibration()
        return self.calibrated

    def _finalize_calibration(self):
        """计算传感器的零偏"""
        self.lin_acc_bias = np.mean(self._lin_acc_samples, axis=0)
        self.gyro_bias = np.mean(self._gyro_samples, axis=0)
        self.calibrated = True
        print("\nIMU 零偏校准完成。")
        print(f"  - 线性加速度零偏 (Lin Acc bias): {self.lin_acc_bias}")
        print(f"  - 陀螺仪零偏 (Gyro bias): {self.gyro_bias}")

    def calculate_delta_pose(self, lin_acc_raw, gyro_raw, dt):
        """
        计算单帧的相对位姿变换。
        返回: (delta_translation, delta_rotation_euler)
        """
        lin_acc_corr = lin_acc_raw - self.lin_acc_bias
        gyro_corr = gyro_raw - self.gyro_bias

        # 应用噪声阈值
        if np.linalg.norm(lin_acc_corr) < self.ACC_THRESHOLD:
            lin_acc_corr[:] = 0.0

        if np.linalg.norm(gyro_corr) < self.GYRO_THRESHOLD:
            gyro_corr[:] = 0.0

        # 计算瞬时位移 (物理上代表 delta_v = a * dt)
        delta_translation = lin_acc_corr * dt

        # 计算瞬时旋转 (角速度 * 时间 -> 欧拉角)
        delta_rotation_euler = gyro_corr * dt

        return delta_translation, delta_rotation_euler


class IPhonePhyphox(BaseTeleopDevice):
    def __init__(self,
                 shm_name, shm_shape, shm_dtype,
                 action_dim=6,
                 action_dtype=np.float64,
                 frequency=30,
                 phyphox_ip="192.168.1.5",
                 phyphox_port=80,
                 calib_samples=150,
                 dt=None,
                 ):
        super().__init__(shm_name, shm_shape, shm_dtype,
                         action_dim=action_dim,
                         action_dtype=action_dtype,
                         frequency=frequency)

        if dt is not None:
            self.dt = dt
            self.frequency = 1.0 / dt
        else:
            self.dt = 1.0 / frequency

        self.url = f"http://{phyphox_ip}:{phyphox_port}/get?lin_accX&lin_accY&lin_accZ&gyroX&gyroY&gyroZ"
        self.processor = IMUProcessor(calib_samples=calib_samples)

    def get_observation(self):
        """获取线加速度和角速度"""
        try:
            response = requests.get(self.url, timeout=0.5)
            response.raise_for_status()
            data = response.json()["buffer"]
            ax, ay, az = data["lin_accX"]["buffer"][-1], data["lin_accY"]["buffer"][-1], data["lin_accZ"]["buffer"][-1]
            gx, gy, gz = data["gyroX"]["buffer"][-1], data["gyroY"]["buffer"][-1], data["gyroZ"]["buffer"][-1]

            if any(v is None for v in [ax, ay, az, gx, gy, gz]):
                return None
            return np.array([ax, ay, az, gx, gy, gz])
        except Exception:
            return None

    def observation_to_action(self, obs):
        """将传感器观测值转换为相对位姿变换"""
        if obs is None:
            return self.get_zero_action()

        lin_acc_raw = obs[:3]
        gyro_raw = obs[3:]

        delta_translation, delta_rotation_euler = self.processor.calculate_delta_pose(
            lin_acc_raw, gyro_raw, self.dt
        )

        return np.concatenate([delta_translation, delta_rotation_euler]).astype(self.action_dtype)

    def run(self):
        """主循环"""
        from deploy.robot.base import RateLimiter
        self.connect_to_buffer()
        rr.init("iphone_imu_relative_pose", spawn=True)
        rr.log("/", rr.ViewCoordinates.RFU, timeless=True)
        rr.log("world/origin_axes", rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=np.eye(3) * 0.1,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ), timeless=True)

        # --- Rerun可视化专用变量 ---
        viz_position = np.zeros(3)
        viz_orientation = R.identity()
        viz_velocity = np.zeros(3)

        rate_limiter = RateLimiter()
        step = 0

        # --- 频率和位姿统计变量 ---
        frame_count = 0
        last_print_time = time.time()

        try:
            print("--- 相对位姿变换捕捉模型 ---")
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
                    self.processor.reset()
                    # 重置可视化位姿
                    viz_position = np.zeros(3)
                    viz_orientation = R.identity()
                    viz_velocity = np.zeros(3)

                observation = self.get_observation()
                if observation is None:
                    rate_limiter.sleep(self.frequency)
                    continue

                if not self.processor.calibrated:
                    print(f"校准中... {self.processor.cnt}/{self.processor.N}", end='\r')
                    lin_acc_raw, gyro_raw = observation[:3], observation[3:]
                    self.processor.calibrate_step(lin_acc_raw, gyro_raw)
                    self.put_action_to_buffer(self.get_zero_action())
                else:
                    # 计算相对变换并写入共享内存
                    delta_pose = self.observation_to_action(observation)
                    self.put_action_to_buffer(delta_pose)

                    # --- 更新Rerun可视化物理模型 ---
                    delta_translation = delta_pose[:3]
                    delta_rotation_euler = delta_pose[3:]

                    # 1. 累积旋转
                    delta_rotation = R.from_euler('xyz', delta_rotation_euler)
                    viz_orientation = viz_orientation * delta_rotation

                    # 2. 将手机坐标系下的平移变换(物理上是Δv)到世界坐标系并累积到速度上
                    world_delta_velocity = viz_orientation.apply(delta_translation)
                    viz_velocity += world_delta_velocity

                    # 3. 应用分轴独立连续动态阻尼以实现平滑制动
                    lin_acc_raw = observation[:3]
                    lin_acc_corr = lin_acc_raw - self.processor.lin_acc_bias
                    world_linear_acc = viz_orientation.apply(lin_acc_corr)

                    gliding_friction = 0.998
                    stopping_friction = 0.85

                    for i in range(3):  # 分别处理 X, Y, Z 轴
                        # 根据当前轴向的加速度大小，连续地计算阻尼影响力
                        stopping_influence = np.clip(1.0 - abs(world_linear_acc[i]) / self.processor.ACC_THRESHOLD, 0,
                                                     1)

                        # 通过影响力，在两种摩擦力之间进行平滑插值
                        damping_factor = (
                                                     1.0 - stopping_influence) * gliding_friction + stopping_influence * stopping_friction

                        # 将计算出的动态阻尼应用到该轴向的速度分量上
                        viz_velocity[i] *= damping_factor

                    # 4. 最终钳制，防止微小蠕动
                    if np.linalg.norm(viz_velocity) < 0.01:
                        viz_velocity[:] = 0

                    # 5. 用更新后的速度更新位置
                    viz_position += viz_velocity * self.dt

                    # --- 打印实时信息 ---
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    if elapsed >= 1.0:
                        frequency_actual = frame_count / elapsed
                        pos_str = f"位置(m): x={viz_position[0]:.2f}, y={viz_position[1]:.2f}, z={viz_position[2]:.2f}"
                        rot_deg = viz_orientation.as_euler('xyz', degrees=True)
                        rot_str = f"旋转(°): r={rot_deg[0]:.1f}, p={rot_deg[1]:.1f}, y={rot_deg[2]:.1f}"
                        print(f"频率: {frequency_actual:.2f} Hz | {pos_str} | {rot_str}      ", end="\r")
                        last_print_time = current_time
                        frame_count = 0

                    # --- Log到Rerun ---
                    step += 1
                    rr.set_time_sequence("step", step)
                    rr.log("iphone", rr.Transform3D(translation=viz_position,
                                                    rotation=rr.Quaternion(xyzw=viz_orientation.as_quat())))
                    rr.log("iphone", rr.Boxes3D(half_sizes=[0.075, 0.035, 0.005]))

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
        IPHONE_IP = "192.168.71.95"
        dev = IPhonePhyphox(
            shm_name=shm_info['name'], shm_shape=shm_info['shape'], shm_dtype=shm_info['dtype'],
            action_dim=action_dim, action_dtype=action_dtype,
            frequency=50,
            phyphox_ip=IPHONE_IP, phyphox_port=80,
            calib_samples=90, dt=0.2
        )
        dev.run()
    finally:
        print("主程序: 清理共享内存。")
        shm.close()
        shm.unlink()
