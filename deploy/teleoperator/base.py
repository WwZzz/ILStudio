import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod

# --- 1. Base Teleoperation Device Class ---

class BaseTeleopDevice(ABC):
    """
    Abstract base class for teleoperation devices
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
        """Connect to existing shared memory"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("Teleop device: Successfully connected to shared memory.")
        except FileNotFoundError:
            print(f"Teleop device: Error! Shared memory '{self.shm_name}' does not exist.")
            raise

    @abstractmethod
    def get_observation(self):
        """Get raw observation data from device (e.g., key states)"""
        pass

    @abstractmethod
    def observation_to_action(self, observation):
        """Convert observation data to standardized robot action"""
        pass

    def put_action_to_buffer(self, action):
        """Write action to shared memory buffer"""
        if self.action_buffer is not None:
            t = time.time()
            self.action_buffer[0]['timestamp'] = t
            self.action_buffer[0]['action'] = action

    def run(self):
        """
        Main loop: get observation, convert to action, and write to buffer at specified frequency
        """
        self.connect_to_buffer()
        rate = 1.0 / self.frequency
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                # Core three steps
                observation = self.get_observation()
                action = self.observation_to_action(observation)
                self.put_action_to_buffer(action)
                elapsed_time = time.time() - start_time
                sleep_time = rate - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            print("Teleop device: Shutting down...")
            if self.shm:
                self.shm.close()

    def stop(self):
        """Set stop event"""
        self.stop_event.set()
        
def str2dtype(s: str):
    # 将参数转换为对应的变量
    if s=='float32' or s=='float': return np.float32
    elif s=='float64' or s=='double': return np.float64
    elif s=='int' or s=='int32': return np.int
    elif s=='long' or s=='int64': return np.int64
    else:
        raise ValueError(f'Invalid string {s}')