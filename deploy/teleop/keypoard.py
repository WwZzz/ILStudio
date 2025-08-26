import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard

class KeyboardTeleop(TeleopDeviceBase):
    """
    使用键盘进行遥操作的具体实现
    """

    def __init__(self, shm_name, shm_shape, shm_dtype, frequency=100):
        super().__init__(shm_name, shm_shape, shm_dtype, frequency)
        self.pressed_keys = set()

        # 将控制灵敏度定义为类内部的属性
        self.MIN_TRANS_STEP = 0.005  # 每次按键的最小平移量 (米)
        self.MIN_ROT_STEP = np.deg2rad(1.5)  # 每次按键的最小旋转量 (弧度)

        self._start_keyboard_listener()

    def _on_press(self, key):
        """按键按下时的回调函数"""
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            # 处理特殊按键，如方向键、空格等
            self.pressed_keys.add(key)

    def _on_release(self, key):
        """按键释放时的回调函数"""
        try:
            self.pressed_keys.remove(key.char)
        except AttributeError:
            self.pressed_keys.remove(key)
        except KeyError:
            pass  # 忽略已经不在集合中的按键

    def _start_keyboard_listener(self):
        """启动一个独立的线程来监听键盘事件"""
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True  # 设置为守护线程，主进程退出时它也退出
        listener.start()
        print("键盘监听器已启动。")

    def get_observation(self):
        """获取当前所有被按下的键"""
        return self.pressed_keys.copy()

    def observation_to_action(self, observation):
        """将按键状态转换为机械臂末端的 delta 位姿和夹爪动作"""
        action = np.zeros(self.shm_shape, dtype=self.shm_dtype)

        # 平移控制 (X, Y, Z)
        if 'a' in observation: action[0] = -self.MIN_TRANS_STEP
        if 'd' in observation: action[0] = self.MIN_TRANS_STEP
        if 'w' in observation: action[1] = self.MIN_TRANS_STEP
        if 's' in observation: action[1] = -self.MIN_TRANS_STEP
        if 'q' in observation: action[2] = self.MIN_TRANS_STEP
        if 'e' in observation: action[2] = -self.MIN_TRANS_STEP

        # 旋转控制 (Roll, Pitch, Yaw)
        if 'j' in observation: action[3] = self.MIN_ROT_STEP
        if 'l' in observation: action[3] = -self.MIN_ROT_STEP
        if 'i' in observation: action[4] = self.MIN_ROT_STEP
        if 'k' in observation: action[4] = -self.MIN_ROT_STEP
        if 'u' in observation: action[5] = self.MIN_ROT_STEP
        if 'o' in observation: action[5] = -self.MIN_ROT_STEP

        # 夹爪控制
        if keyboard.Key.space in observation:
            action[6] = -1.0  # 闭合信号
        else:
            action[6] = 1.0  # 张开信号

        return action