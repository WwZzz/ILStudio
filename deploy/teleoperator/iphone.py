import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from deploy.teleoperator.base import BaseTeleopDevice, str2dtype   # 你的基类

class IMUZeroCorr:
    """
    零位校正 + 位移增量 + 四元数旋转增量
    acc_unit : m/s²
    gyro_unit: rad/s
    """
    def __init__(self, calib_samples: int = 200):
        self.N = calib_samples
        self.cnt = 0
        self.acc_bias  = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.calibrated = False

    # -------------------------------------------------
    def feed(self, acc, gyro):
        """喂一帧原始数据，返回 (dxyz, delta_q) 或 None（标定阶段）"""
        acc  = np.asarray(acc,  float)
        gyro = np.asarray(gyro, float)

        # 1) 标定阶段
        if not self.calibrated:
            self.acc_bias  += acc
            self.gyro_bias += gyro
            self.cnt += 1
            if self.cnt >= self.N:
                self.acc_bias  /= self.N
                self.gyro_bias /= self.N
                self.calibrated = True
                print("零偏标定完成")
                print("acc bias :", self.acc_bias)
                print("gyro bias:", self.gyro_bias)
            return None, None

        # 2) 零偏补偿
        acc_corr  = acc  - self.acc_bias
        gyro_corr = gyro - self.gyro_bias

        return acc_corr, gyro_corr

    # -------------------------------------------------
    def delta(self, acc_corr, gyro_corr, dt):
        """输入零偏补偿后的 acc/gyro，返回 dxyz、delta_q"""
        # 位移增量
        dxyz = 0.5 * acc_corr * dt**2

        # 角度增量 → 四元数
        dtheta = gyro_corr * dt
        angle  = np.linalg.norm(dtheta)
        if angle < 1e-12:
            delta_q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = dtheta / angle
            s = np.sin(angle / 2.0)
            delta_q = np.array([np.cos(angle / 2.0),
                                axis[0] * s,
                                axis[1] * s,
                                axis[2] * s])
        return dxyz, delta_q

    # -------------------------------------------------
    def reset_bias(self):
        """手动重新标定"""
        self.cnt = 0
        self.acc_bias[:]  = 0
        self.gyro_bias[:] = 0
        self.calibrated = False
        print("已重置零偏，请保持手机静止…")



class IPhonePhyphox(BaseTeleopDevice):
    def __init__(self,
                 shm_name, shm_shape, shm_dtype,
                 action_dim=6,
                 phyphox_ip="192.168.1.5",
                 phyphox_port=8080,
                 calib_samples=200,
                 frequency=100):
        super().__init__(shm_name, shm_shape, str2dtype(shm_dtype),
                         action_dim=action_dim,
                         frequency=frequency)
        self.url = f"http://{phyphox_ip}:{phyphox_port}/get?accX&accY&accZ&gyroX&gyroY&gyroZ&"
        self.imu = IMUZeroCorr(calib_samples=calib_samples)
        self.dt = 1.0 / frequency
        self.prev_time = None

    # -------------------------------------------------
    def get_observation(self):
        """
        从 Phyphox 获取原始 acc(m/s²) 与 gyro(rad/s)
        返回 (ax, ay, az, gx, gy, gz) 或 None 如果请求失败
        """
        try:
            r = requests.get(self.url,
                             params=["accX", "accY", "accZ",
                                     "gyrX", "gyrY", "gyrZ"],
                             timeout=0.1)
            data = r.json()["buffer"]
            # 取最新采样点
            ax, ay, az = [data[k]["data"][-1] for k in ("accX", "accY", "accZ")]
            gx, gy, gz = [data[k]["data"][-1] for k in ("gyrX", "gyrY", "gyrZ")]
            return np.array([ax, ay, az, gx, gy, gz])
        except Exception as e:
            print("Phyphox read error:", e)
            return None

    # -------------------------------------------------
    def observation_to_action(self, obs):
        """
        obs: [ax, ay, az, gx, gy, gz]
        返回: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]  (m, rad)
        """
        if obs is None:
            return self.get_zero_action()

        acc_raw, gyro_raw = obs[:3], obs[3:]
        acc_corr, gyro_corr = self.imu.feed(acc_raw, gyro_raw)

        if acc_corr is None:
            # 仍在零偏标定阶段，输出零动作
            return self.get_zero_action()

        # 1) 位移增量
        dxyz, delta_q = self.imu.delta(acc_corr, gyro_corr, self.dt)

        # 2) 四元数 → 欧拉角
        delta_rot = R.from_quat(delta_q[[1, 2, 3, 0]])  # [x,y,z,w] for scipy
        d_euler = delta_rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]

        action = np.concatenate([dxyz, d_euler]).astype(self.action_dtype)
        return action