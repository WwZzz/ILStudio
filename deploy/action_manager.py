import collections
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import time

# 类型别名，方便阅读
ActionChunk = np.ndarray
ActionStep = np.ndarray
BufferItem = Tuple[float, ActionChunk]


class ActionChunkBuffer(ABC):
    """
    统一管理聚合数组的动作处理管道基类。

    所有子类都将继承 _latest_agg_chunk 及其相关的滑动窗口和时间戳管理。
    子类只需实现 _populate_agg_chunk 方法来定义填充聚合数组的逻辑。
    """

    def __init__(self, max_chunk_count: int, chunk_length: int, frequency: float):
        if chunk_length <= 0 or frequency <= 0:
            raise ValueError("chunk_length and frequency must be positive.")

        self.max_chunk_count = max_chunk_count
        self.chunk_length = chunk_length
        self.frequency = frequency
        self._dt = 1.0 / self.frequency

        self._buffer: collections.deque = collections.deque(maxlen=self.max_chunk_count)
        self._condition: threading.Condition = threading.Condition()

        # 聚合数组的固定长度
        self.agg_length = self.max_chunk_count * self.chunk_length

        # 预分配的固定大小的聚合数组
        self._latest_agg_chunk: Optional[ActionChunk] = None

        # 聚合数组的时间戳管理
        self._agg_start_timestamp: Optional[float] = None
        self._agg_end_timestamp: Optional[float] = None

    def put(self, timestamp: float, action_chunk: ActionChunk):
        """将一个新的动作块放入缓冲区，并进行清理、内部状态更新和通知。"""
        if not isinstance(action_chunk, np.ndarray) or action_chunk.shape[0] != self.chunk_length:
            raise ValueError(f"action_chunk must be a numpy array of shape ({self.chunk_length}, ...)")

        with self._condition:
            # 清理过时动作块
            while self._buffer:
                oldest_chunk_timestamp, _ = self._buffer[0]
                oldest_chunk_end_time = oldest_chunk_timestamp + (self.chunk_length - 1) * self._dt

                if oldest_chunk_end_time < timestamp:
                    self._buffer.popleft()
                else:
                    break

            # 添加新动作块
            self._buffer.append((timestamp, action_chunk))

            # 刷新聚合数组
            self._internal_update()

            # 通知等待的线程
            self._condition.notify_all()

    def get(self, current_timestamp: float) -> ActionStep:
        """根据当前时间戳获取一个动作步，如果时间戳在未来则阻塞等待。"""
        with self._condition:
            while not self._check_time_condition(current_timestamp):
                self._condition.wait()

            if not self._is_timestamp_valid(current_timestamp):
                raise RuntimeError(f"在等待后，时间 {current_timestamp:.3f} 依然超出有效范围。")

            return self._get_action_from_agg_chunk(current_timestamp)

    def _internal_update(self):
        """
        统一的内部更新逻辑，由基类管理。
        """
        if not self._buffer:
            self._latest_agg_chunk = None
            self._agg_start_timestamp = None
            self._agg_end_timestamp = None
            return

        oldest_ts, _ = self._buffer[0]
        newest_ts, newest_chunk = self._buffer[-1]

        # 初始化或重建聚合数组
        if self._latest_agg_chunk is None:
            self._latest_agg_chunk = np.zeros((self.agg_length, newest_chunk.shape[1]), dtype=np.float32)
            self._agg_start_timestamp = oldest_ts

        # 检查时间窗口是否需要滑动
        ts_shift = oldest_ts - self._agg_start_timestamp
        if abs(ts_shift) > 1e-9:
            idx_shift = int(round(ts_shift / self._dt))
            self._latest_agg_chunk = np.roll(self._latest_agg_chunk, -idx_shift, axis=0)
            if idx_shift > 0:
                self._latest_agg_chunk[-idx_shift:] = 0
            self._agg_start_timestamp = oldest_ts

        self._agg_end_timestamp = self._agg_start_timestamp + (self.agg_length - 1) * self._dt

        # 调用子类方法填充聚合数组
        self._populate_agg_chunk()

    def _get_action_from_agg_chunk(self, current_timestamp: float) -> ActionStep:
        """从聚合数组中获取动作，这是一个 O(1) 的操作。"""
        start_time = self._agg_start_timestamp
        action_index = int(round((current_timestamp - start_time) / self._dt))
        safe_index = np.clip(action_index, 0, self.agg_length - 1)
        return self._latest_agg_chunk[safe_index]

    def _check_time_condition(self, current_timestamp: float) -> bool:
        """检查最新动作块的开始时间是否已不早于请求时间。"""
        if not self._buffer:
            return False
        latest_timestamp, _ = self._buffer[-1]
        return latest_timestamp <= current_timestamp

    def _is_timestamp_valid(self, current_timestamp: float) -> bool:
        """检查时间戳是否在当前聚合数组的有效时间范围内。"""
        if self._agg_start_timestamp is None:
            return False
        return self._agg_start_timestamp <= current_timestamp <= self._agg_end_timestamp

    @abstractmethod
    def _populate_agg_chunk(self):
        """抽象方法：由子类实现，用于定义如何填充聚合数组。"""
        pass


class LatestActionChunkBuffer(ActionChunkBuffer):
    def _populate_agg_chunk(self):
        """
        填充聚合数组：从最新到最旧，将缓冲区中的动作块覆盖到聚合数组中。
        """
        for source_timestamp, source_chunk in reversed(self._buffer):
            for i in range(self.chunk_length):
                source_time = source_timestamp + i * self._dt
                agg_index = int(round((source_time - self._agg_start_timestamp) / self._dt))

                if 0 <= agg_index < self.agg_length:
                    self._latest_agg_chunk[agg_index] = source_chunk[i]


class WeightedAverageActionChunkBuffer(ActionChunkBuffer):
    def __init__(self, max_chunk_count: int, chunk_length: int, frequency: float, decay_factor: float = 1.0):
        super().__init__(max_chunk_count, chunk_length, frequency)
        self.decay_factor = decay_factor

    def _populate_agg_chunk(self):
        """
        填充聚合数组：对所有重叠的动作进行加权平均。
        """
        newest_ts, newest_chunk = self._buffer[-1]

        for i in range(self.agg_length):
            target_time = self._agg_start_timestamp + i * self._dt

            total_weighted_action = np.zeros_like(newest_chunk[0], dtype=np.float32)
            total_weight = 0.0

            for source_timestamp, source_chunk in self._buffer:
                source_start_time = source_timestamp
                source_end_time = source_timestamp + (self.chunk_length - 1) * self._dt

                if source_start_time <= target_time <= source_end_time:
                    source_index = int(round((target_time - source_timestamp) / self._dt))
                    safe_source_index = np.clip(source_index, 0, self.chunk_length - 1)
                    action_to_aggregate = source_chunk[safe_source_index]

                    time_diff = newest_ts - source_timestamp
                    weight = np.exp(-self.decay_factor * time_diff)

                    total_weighted_action += action_to_aggregate * weight
                    total_weight += weight

            if total_weight > 0:
                self._latest_agg_chunk[i] = total_weighted_action / total_weight
            else:
                # 备用：如果该时间点无重叠动作，则使用最新的动作作为备用
                latest_chunk_index = int(round((target_time - newest_ts) / self._dt))
                safe_latest_index = np.clip(latest_chunk_index, 0, self.chunk_length - 1)
                self._latest_agg_chunk[i] = newest_chunk[safe_latest_index]