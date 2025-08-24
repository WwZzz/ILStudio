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


# import threading
# import time
# import numpy as np


# # 假设上述 ActionChunkBuffer 和其子类代码都已在当前文件中

# def test_producer(buffer: ActionChunkBuffer, put_rate: float, num_puts: int, chunk_len: int, freq: float):
#     """模拟一个策略线程，以固定速率生成动作块并put入缓冲区。"""
#     print(f"[生产者] 启动，每 {put_rate:.3f} 秒生成一个动作块。")
#     start_time = time.time()

#     for i in range(num_puts):
#         current_time = (time.time() - start_time)
#         action_value = i + 1  # 使用不同的值来区分每个动作块
#         action_chunk = np.array([
#             [action_value, current_time + j * (1.0 / freq)] for j in range(chunk_len)
#         ], dtype=np.float32)

#         buffer.put(current_time, action_chunk)
#         print(f"[生产者] 在时间 {current_time:.3f} 放入了第 {i + 1} 个动作块 (值: {action_value:.1f})。")
#         time.sleep(put_rate)
#     print("[生产者] 任务完成。")


# def test_consumer(buffer: ActionChunkBuffer, get_rate: float, num_gets: int, start_offset: float):
#     """模拟一个机器人控制线程，以固定速率从缓冲区获取动作。"""
#     print(f"[消费者] 启动，每 {get_rate:.3f} 秒尝试获取一个动作。")
#     start_time = time.time()

#     for i in range(num_gets):
#         current_time = (time.time() - start_time) + start_offset
#         print(f"[消费者] 尝试在时间 {current_time:.3f} 获取动作...")
#         try:
#             action = buffer.get(current_time)
#             print(f"[消费者] 成功获取动作: {action[0]:.3f} @ {current_time:.3f}")
#         except RuntimeError as e:
#             print(f"[消费者] 错误: {e}")
#         time.sleep(get_rate)


# def run_tests():
#     MAX_CHUNKS = 3
#     CHUNK_LENGTH = 10
#     FREQUENCY = 20

#     # ----------------------------------------------------
#     # Test Case 1: LatestActionChunkBuffer - 线程同步与数据覆盖
#     # ----------------------------------------------------
#     print("\n" + "=" * 50)
#     print("测试1: LatestActionChunkBuffer - 线程同步与数据覆盖")
#     print("=" * 50)
#     latest_buffer = LatestActionChunkBuffer(MAX_CHUNKS, CHUNK_LENGTH, FREQUENCY)

#     producer_thread_latest = threading.Thread(
#         target=test_producer, args=(latest_buffer, 0.5, 5, CHUNK_LENGTH, FREQUENCY)
#     )
#     consumer_thread_latest = threading.Thread(
#         target=test_consumer, args=(latest_buffer, 0.2, 10, 0.0)
#     )

#     producer_thread_latest.start()
#     consumer_thread_latest.start()

#     producer_thread_latest.join()
#     consumer_thread_latest.join()

#     print("=> 验证结果:")
#     print("=> 消费者应在开始时阻塞，然后获取的动作值应始终是最新的那个chunk的值 (如 1.0, 2.0, 3.0...)")

#     # ----------------------------------------------------
#     # Test Case 2: WeightedAverageActionChunkBuffer - 加权平均与滑动窗口
#     # ----------------------------------------------------
#     print("\n" + "=" * 50)
#     print("测试2: WeightedAverageActionChunkBuffer - 加权平均与滑动窗口")
#     print("=" * 50)
#     weighted_buffer = WeightedAverageActionChunkBuffer(MAX_CHUNKS, CHUNK_LENGTH, FREQUENCY, decay_factor=5.0)

#     producer_thread_weighted = threading.Thread(
#         target=test_producer, args=(weighted_buffer, 0.2, 5, CHUNK_LENGTH, FREQUENCY)
#     )
#     consumer_thread_weighted = threading.Thread(
#         target=test_consumer, args=(weighted_buffer, 0.2, 10, 0.0)
#     )

#     producer_thread_weighted.start()
#     consumer_thread_weighted.start()

#     producer_thread_weighted.join()
#     consumer_thread_weighted.join()

#     print("=> 验证结果:")
#     print("=> 消费者获取的动作值应是多个chunk的加权平均值，会是一个平滑的过渡值。")
#     print("=> 例如，在chunk 1和2重叠时，动作值会介于1.0和2.0之间。")

#     # ----------------------------------------------------
#     # Test Case 3: 缓冲区清理逻辑 (非线程)
#     # ----------------------------------------------------
#     print("\n" + "=" * 50)
#     print("测试3: 缓冲区清理逻辑")
#     print("=" * 50)
#     buffer = LatestActionChunkBuffer(MAX_CHUNKS, CHUNK_LENGTH, FREQUENCY)
#     dt = 1.0 / FREQUENCY

#     print(f"初始缓冲区大小: {len(buffer._buffer)}")

#     # put 第一个chunk
#     t1 = 0.0
#     action_chunk_1 = np.array([[1.0, 0.0]] * CHUNK_LENGTH)
#     buffer.put(t1, action_chunk_1)
#     print(f"put chunk @ {t1:.3f}s. 缓冲区大小: {len(buffer._buffer)}. agg_start: {buffer._agg_start_timestamp:.3f}")

#     # put 第二个chunk，时间重叠
#     t2 = 0.1
#     action_chunk_2 = np.array([[2.0, 0.0]] * CHUNK_LENGTH)
#     buffer.put(t2, action_chunk_2)
#     print(
#         f"put chunk @ {t2:.3f}s (重叠). 缓冲区大小: {len(buffer._buffer)}. agg_start: {buffer._agg_start_timestamp:.3f}")

#     # put 第三个chunk，时间重叠
#     t3 = 0.2
#     action_chunk_3 = np.array([[3.0, 0.0]] * CHUNK_LENGTH)
#     buffer.put(t3, action_chunk_3)
#     print(
#         f"put chunk @ {t3:.3f}s (重叠). 缓冲区大小: {len(buffer._buffer)}. agg_start: {buffer._agg_start_timestamp:.3f}")

#     # put 第四个chunk，超过 max_chunks 限制，但时间不重叠
#     t4 = 2.0  # 确保早于t1+chunk_len*dt的结束时间
#     action_chunk_4 = np.array([[4.0, 0.0]] * CHUNK_LENGTH)
#     buffer.put(t4, action_chunk_4)
#     print(
#         f"put chunk @ {t4:.3f}s (不重叠). 缓冲区大小: {len(buffer._buffer)}. agg_start: {buffer._agg_start_timestamp:.3f}")

#     # 最后一个 chunk (t4) 的结束时间是 2.0 + 10*0.05 = 2.5s
#     # put 第五个 chunk，其开始时间大于 t1 的结束时间，但小于 t2, t3, t4 的结束时间
#     t5 = 1.0
#     action_chunk_5 = np.array([[5.0, 0.0]] * CHUNK_LENGTH)
#     buffer.put(t5, action_chunk_5)
#     print(f"put chunk @ {t5:.3f}s. 缓冲区大小: {len(buffer._buffer)}. agg_start: {buffer._agg_start_timestamp:.3f}")

#     print("=> 验证结果:")
#     print("=> 缓冲区大小应始终 <= 3。")
#     print("=> 经过put(t=2.0)后，t1, t2, t3的结束时间都小于2.0，因此缓冲区应只剩下t4一个chunk。")
#     print(f"   此时缓冲区内容: {[t for t, _ in buffer._buffer]}")
#     print("=> 经过put(t=1.0)后，缓冲区应包含t5和t4。")
#     print(f"   此时缓冲区内容: {[t for t, _ in buffer._buffer]}")


# if __name__ == "__main__":
#     run_tests()