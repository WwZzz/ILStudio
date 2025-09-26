import collections
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np
import time
import queue

def load_action_manager(manager_name: str, manager_config: Any):
    try: 
        MANAGER_CLASS = eval(manager_name)
    except Exception as e:
        print(f"Failed to load manager class {manager_name}")
    manager = MANAGER_CLASS(manager_config)
    return manager

class AbstractActionManager(ABC):
    @abstractmethod
    def put(self, chunk: np.ndarray, timestamp: float=None):
        """Put action chunk into local cache"""
        pass

    @abstractmethod
    def get(self, timestamp: float=None):
        """Get one-step action from local cache"""
        pass

class BasicActionManager(AbstractActionManager):
    """Drop out the previous chunk directly whenever the new one comes"""
    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self.current_step = 0
        self._buffer = None
        self._chunk_buffer = None

    def put(self, chunk, timestamp:float=None):
        with self._lock:
            self._chunk_buffer = chunk
            self.current_step = 0

    def get(self, timestamp: float=None):
        with self._lock:
            if self._chunk_buffer is None or self.current_step>=len(self._chunk_buffer):
                return None
            action = self._chunk_buffer[self.current_step]
            self.current_step += 1
            return action

class OlderFirstManager(BasicActionManager):
    """Refuse newly coming chunks unless the last chunk ends x%"""
    def __init__(self, config):
        super().__init__(config)
        self.coef = getattr(config, 'manager_coef', 1.0)

    def put(self, chunk, timestamp:float=None):
        if self._chunk_buffer is None:
            super().put(chunk, timestamp)
        else:
            with self._lock:
                if self.current_step < int(len(self._chunk_buffer)*self.coef):
                    return
            super().put(chunk, timestamp)



class TemporalAggManager(BasicActionManager):
    """Expotionally average the last and the new chunks for better smoothness"""
    def __init__(self, config):
        super().__init__(config)
        self.coef = getattr(config, 'manager_coef', 0.1)

    def put(self, chunk, timestamp:float=None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            with self._lock:
                prev_step = self.current_step
                prev_len = len(self._chunk_buffer)
                remain_len = prev_len - prev_step
                if remain_len>0:
                    for idx in range(remain_len):
                        chunk[idx]['action'] = (1.-self.coef) * chunk[idx]['action'] + self.coef*self._chunk_buffer[idx+prev_step]['action']
                self._chunk_buffer = chunk
                self.current_step = 0


class TemporalOlderManager(TemporalAggManager):
    """Refuse newly coming chunks until the last chunk ends x%"""
    def __init__(self, config):
        super().__init__(config)
        self.older_coef = 0.75

    def put(self, chunk, timestamp:float=None):
        if self._chunk_buffer is None:
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
        else:
            with self._lock:
                if self.current_step < int(len(self._chunk_buffer)*self.older_coef):
                    return
            super().put(chunk, timestamp)

class DelayFreeManager(BasicActionManager):
    """Remove the outdated ations from each chunk"""
    def __init__(self, config):
        super().__init__(config)
        self.duration = getattr(config, 'manager_coef', 0.05)


    def put(self, chunk, timestamp:float=None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            delay_time = time.perf_counter() - timestamp
            delayed_start_idx = int(delay_time//self.duration)

            if delayed_start_idx<len(chunk):
                chunk = chunk[delayed_start_idx:]
                with self._lock:
                    self._chunk_buffer = chunk
                    self.current_step = 0
