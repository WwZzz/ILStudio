import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod

# --- 1. 遥操作设备基类 ---

class BaseTeleopDevice(ABC):
    """
    遥操作设备抽象基类
    """
    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=100):
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.frequency = frequency
        self.shm = None
        self.action_buffer = None
        self.stop_event = mp.Event()

    def connect_to_buffer(self):
        """连接到已存在的共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("遥操作设备：成功连接到共享内存。")
        except FileNotFoundError:
            print(f"遥操作设备：错误！共享内存 '{self.shm_name}' 不存在。")
            raise

    @abstractmethod
    def get_observation(self):
        """从设备获取原始观测数据（例如，按键状态）"""
        pass

    @abstractmethod
    def observation_to_action(self, observation):
        """将观测数据转换为标准化的机器人动作"""
        pass

    def put_action_to_buffer(self, action):
        """将动作写入共享内存缓冲区"""
        if self.action_buffer is not None:
            self.action_buffer[:] = action

    def run(self):
        """
        主循环，以指定频率获取观测、转换动作并写入缓冲区
        """
        self.connect_to_buffer()
        rate = 1.0 / self.frequency
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                # 核心三步
                observation = self.get_observation()
                action = self.observation_to_action(observation)
                self.put_action_to_buffer(action)
                elapsed_time = time.time() - start_time
                sleep_time = rate - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("遥操作设备：正在关闭...")
            if self.shm:
                self.shm.close()

    def stop(self):
        """设置停止事件"""
        self.stop_event.set()


# --- 2. 键盘遥操作实现 ---



# --- 3. 机器人控制器 ---

class MockRobot:
    """一个模拟的机器人接口，用于演示"""

    def __init__(self):
        self.gripper_width = 0.1  # 初始夹爪宽度
        self.max_gripper_width = 0.1
        self.min_gripper_width = 0.0

    def publish_action(self, action):
        """
        模拟发布动作到机器人。
        在真实场景中，这里会调用机器人的API。
        """
        gripper_signal = action[6]
        gripper_speed = 0.01  # 模拟夹爪开合速度
        self.gripper_width += gripper_signal * gripper_speed * 0.1  # 乘以一个小的 dt
        self.gripper_width = np.clip(self.gripper_width, self.min_gripper_width, self.max_gripper_width)

        trans_part = f"Pos: [{action[0]: .4f}, {action[1]: .4f}, {action[2]: .4f}]"
        rot_part = f"Rot: [{action[3]: .4f}, {action[4]: .4f}, {action[5]: .4f}]"
        gripper_part = f"Gripper: {self.gripper_width:.3f}"

        print(f"\r发布动作 -> {trans_part} | {rot_part} | {gripper_part}", end="")


class RobotController:
    """
    从共享内存读取动作并发送给机器人
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=200):
        self.robot = MockRobot()
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.frequency = frequency
        self.shm = None
        self.action_buffer = None
        self.stop_event = mp.Event()

    def connect_to_buffer(self):
        """连接到共享内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("机器人控制器：成功连接到共享内存。")
        except FileNotFoundError:
            print(f"机器人控制器：错误！共享内存 '{self.shm_name}' 不存在。")
            raise

    def run(self):
        """
        主循环，非阻塞地从缓冲区获取数据并发送给机器人
        """
        self.connect_to_buffer()
        rate = 1.0 / self.frequency
        try:
            while not self.stop_event.is_set():
                start_time = time.time()

                action = self.action_buffer.copy()
                self.robot.publish_action(action)

                elapsed_time = time.time() - start_time
                sleep_time = rate - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("\n机器人控制器：正在关闭...")
            if self.shm:
                self.shm.close()

    def stop(self):
        self.stop_event.set()


# --- 4. 主程序入口 ---

def main():
    # --- 实时配置参数 ---
    ACTION_SHM_NAME = 'teleop_action_buffer'
    ACTION_DIM = 7
    ACTION_DTYPE = np.float64
    TELEOP_FREQUENCY = 100  # Hz
    ROBOT_FREQUENCY = 200  # Hz

    # 定义共享内存的规格
    shm_shape = (ACTION_DIM,)
    shm_size = np.prod(shm_shape) * np.dtype(ACTION_DTYPE).itemsize

    # 创建共享内存块
    try:
        shm = shared_memory.SharedMemory(name=ACTION_SHM_NAME, create=True, size=shm_size)
        print(f"主程序：成功创建共享内存 '{ACTION_SHM_NAME}'，大小 {shm_size} 字节。")
    except FileExistsError:
        print(f"主程序：共享内存 '{ACTION_SHM_NAME}' 已存在，将连接到它。")
        shm = shared_memory.SharedMemory(name=ACTION_SHM_NAME)

    # 初始化遥操作设备和机器人控制器，并传入配置
    teleop_device = KeyboardTeleop(
        shm_name=ACTION_SHM_NAME,
        shm_shape=shm_shape,
        shm_dtype=ACTION_DTYPE,
        frequency=TELEOP_FREQUENCY
    )
    robot_controller = RobotController(
        shm_name=ACTION_SHM_NAME,
        shm_shape=shm_shape,
        shm_dtype=ACTION_DTYPE,
        frequency=ROBOT_FREQUENCY
    )

    # 为每个组件创建一个独立的进程
    teleop_process = mp.Process(target=teleop_device.run)
    robot_process = mp.Process(target=robot_controller.run)

    try:
        print("\n--- 控制说明 ---")
        print("平移: W/S (前/后), A/D (左/右), Q/E (上/下)")
        print("旋转: U/O (绕X轴), I/K (绕Y轴), J/L (绕Z轴)")
        print("夹爪: 按住 空格键 闭合, 松开 张开")
        print("按 Ctrl+C 退出程序。")
        print("-----------------\n")

        teleop_process.start()
        robot_process.start()

        teleop_process.join()
        robot_process.join()

    except KeyboardInterrupt:
        print("\n主程序：检测到 Ctrl+C，正在请求所有进程停止...")
    finally:
        teleop_device.stop()
        robot_controller.stop()

        if teleop_process.is_alive():
            teleop_process.join()
        if robot_process.is_alive():
            robot_process.join()

        print("主程序：所有进程已停止。")

        shm.close()
        shm.unlink()
        print(f"主程序：共享内存 '{ACTION_SHM_NAME}' 已被清理。")


if __name__ == '__main__':
    main()
