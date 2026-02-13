import numpy as np
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from deploy.inference import InferenceContext


class AbstractActionManager(ABC):
    """Abstract base class for action managers.
    
    Action managers manage the buffering and aggregation of action chunks
    from policy inference. They support:
    - select_action(): Main interface - get single-step action,
      triggering inference when buffer is empty or should_infer() returns True.
    - put(): Store a new action chunk into the buffer
    - get(): Retrieve a single-step action from the buffer
    - should_infer(): Determine if new inference is needed (beyond buffer-empty condition)
    - reset(): Reset the action manager state
    
    The step counter `t` is maintained internally and auto-incremented on each
    select_action() call. Subclasses can access it via `self.t`.
    """
    
    @abstractmethod
    def select_action(self):
        """
        Get single-step action, triggering inference if needed.
        
        No parameters — the step counter is maintained internally (self.t).
        Obs is written to SHM externally (sim) or read by the worker (real).
            
        Returns:
            Single-step action
        """
        pass
    
    @abstractmethod
    def put(self, chunk: np.ndarray, timestamp: float = None):
        """Put action chunk into local cache"""
        pass

    @abstractmethod
    def get(self, timestamp: float = None):
        """Get one-step action from local cache"""
        pass

    @abstractmethod
    def should_infer(self) -> bool:
        """
        User-overridable condition for triggering inference.
        
        This is checked IN ADDITION to the buffer-empty condition.
        When this returns True, a new inference will be triggered even if
        the buffer still has actions to dispatch.
        
        Subclasses can use self.t (step count) or time.perf_counter()
        for their decisions.
        
        Default returns False, meaning inference only happens when buffer is empty,
        which naturally forms synchronous execution.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the action manager state."""
        pass


class BasicActionManager(AbstractActionManager):
    """
    Basic action manager: drops the previous chunk whenever a new one arrives.
    
    Inference is always delegated to an inference_worker subprocess via
    InferenceContext (SHM). The flow is:
    
    1. Poll SHM for new action chunks from inference subprocess
    2. If buffer empty or should_infer(): send trigger to subprocess via ctrl_shm
    3. If buffer empty: block-wait for inference result
    4. Return single-step action via get()
    
    The action_manager never touches obs — obs is handled externally:
    - Sim: main process writes obs to obs_shm
    - Real: inference worker reads device SHMs directly
    """
    
    def __init__(self, debug: bool = False, **kwargs):
        self._lock = threading.Lock()
        self.t = 0              # Step counter, auto-incremented
        self.current_step = 0
        self._buffer = None
        self._chunk_buffer = None
        self._inference_ctx: Optional["InferenceContext"] = None
        self._debug = debug
        
        # Debug statistics
        self._total_chunks_received = 0
        self._total_chunks_discarded = 0  # Refused by put (e.g. OlderFirst)
        self._total_triggers_sent = 0
        self._total_waits = 0
        self._cumulative_wait_time = 0.0
        
        if self._debug:
            logger.info(f"[ActionManager] {self.__class__.__name__} initialized in DEBUG mode")

    def set_inference_context(self, inference_ctx: "InferenceContext"):
        """Set inference context for SHM-based inference."""
        self._inference_ctx = inference_ctx

    def select_action(self):
        """
        Get single-step action, triggering inference if needed.
        
        Flow:
        1. Poll for new action chunks from inference subprocess (non-blocking)
        2. If buffer empty or should_infer(): send trigger to subprocess via ctrl_shm
        3. If buffer empty: block-wait for inference result
        4. Return single-step action via get()
        5. Auto-increment self.t
        """
        if self._inference_ctx is None:
            raise RuntimeError("InferenceContext not set. Call set_inference_context() first.")
        
        t_select_start = time.perf_counter() if self._debug else None
        
        # Step 1: Poll for new action chunks (non-blocking)
        mact_list = self._inference_ctx.poll_action_chunks()
        if mact_list:
            chunk_len = len(mact_list)
            self._total_chunks_received += 1
            if self._debug:
                buf_before = self._buffer_status()
                logger.debug(
                    f"[ActionManager] t={self.t} | Chunk #{self._total_chunks_received} arrived | "
                    f"chunk_len={chunk_len} | buffer_before: {buf_before}"
                )
            self.put(mact_list, timestamp=time.perf_counter())
            if self._debug:
                buf_after = self._buffer_status()
                logger.debug(
                    f"[ActionManager] t={self.t} | After put: {buf_after}"
                )
        
        # Step 2: Check if we need to trigger inference
        buffer_empty = self.is_empty()
        need_infer = self.should_infer()
        if buffer_empty or need_infer:
            self._inference_ctx.send_trigger(self.t)
            self._total_triggers_sent += 1
            if self._debug:
                reason = "buffer_empty" if buffer_empty else "should_infer"
                logger.debug(
                    f"[ActionManager] t={self.t} | Trigger sent (reason={reason}) | "
                    f"trigger_count={self._total_triggers_sent}"
                )
        
        # Step 3: If buffer empty, block-wait for inference result
        if buffer_empty:
            if self._debug:
                logger.debug(f"[ActionManager] t={self.t} | Buffer empty, waiting for chunk...")
            self._wait_for_action_chunk(timeout=360.0)
        
        # Step 4: Get single-step action
        action = self.get()
        if action is None:
            raise RuntimeError("ActionManager returned None - no action available")
        
        # Step 5: Auto-increment step counter
        self.t += 1
        
        if self._debug:
            elapsed_ms = (time.perf_counter() - t_select_start) * 1000
            logger.debug(
                f"[ActionManager] t={self.t - 1} → t={self.t} | "
                f"select_action took {elapsed_ms:.2f}ms | {self._buffer_status()}"
            )
        
        return action

    def _wait_for_action_chunk(self, timeout: float = 30.0):
        """Block until a new action chunk is available."""
        t_start = time.perf_counter()
        self._total_waits += 1
        
        while self.is_empty():
            mact_list = self._inference_ctx.poll_action_chunks()
            if mact_list:
                wait_ms = (time.perf_counter() - t_start) * 1000
                self._cumulative_wait_time += wait_ms
                self._total_chunks_received += 1
                if self._debug:
                    logger.debug(
                        f"[ActionManager] t={self.t} | Wait done: {wait_ms:.2f}ms | "
                        f"chunk #{self._total_chunks_received} len={len(mact_list)} | "
                        f"avg_wait={self._cumulative_wait_time / self._total_waits:.2f}ms"
                    )
                self.put(mact_list, timestamp=time.perf_counter())
                break
            
            elapsed = time.perf_counter() - t_start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout ({timeout}s) waiting for inference result. "
                    f"Check if inference process is alive."
                )
            
            if not self._inference_ctx.is_alive():
                raise RuntimeError("Inference process died unexpectedly")
            
            time.sleep(0.0001)  # 0.1ms

    def put(self, chunk, timestamp: float = None):
        with self._lock:
            old_len = len(self._chunk_buffer) if self._chunk_buffer is not None else 0
            old_step = self.current_step
            self._chunk_buffer = chunk
            self.current_step = 0
            if self._debug:
                new_len = len(chunk) if chunk is not None else 0
                remained = max(0, old_len - old_step) if old_len > 0 else 0
                logger.debug(
                    f"[ActionManager] t={self.t} | PUT: new_chunk_len={new_len} | "
                    f"replaced_buffer (was {old_step}/{old_len}, {remained} remaining)"
                )

    def get(self, timestamp: float = None):
        with self._lock:
            if self._chunk_buffer is None or self.current_step >= len(self._chunk_buffer):
                if self._debug:
                    logger.debug(f"[ActionManager] t={self.t} | GET: buffer exhausted → None")
                return None
            action = self._chunk_buffer[self.current_step]
            self.current_step += 1
            if self._debug:
                remaining = len(self._chunk_buffer) - self.current_step
                logger.debug(
                    f"[ActionManager] t={self.t} | GET: step {self.current_step}/{len(self._chunk_buffer)} | "
                    f"remaining={remaining}"
                )
            return action

    def is_empty(self) -> bool:
        with self._lock:
            if self._chunk_buffer is None:
                return True
            return self.current_step >= len(self._chunk_buffer)

    def should_infer(self) -> bool:
        """Default: always False → inference only when buffer is empty (synchronous)."""
        return False

    def _buffer_status(self) -> str:
        """Return a human-readable string describing current buffer state."""
        with self._lock:
            if self._chunk_buffer is None:
                return "buffer=None"
            total = len(self._chunk_buffer)
            consumed = self.current_step
            remaining = total - consumed
            return f"buffer={consumed}/{total} (remaining={remaining})"

    def reset(self):
        with self._lock:
            self._chunk_buffer = None
            self._buffer = None
            self.current_step = 0
            self.t = 0
        if self._inference_ctx is not None:
            epoch = self._inference_ctx._epoch + 1  # Will be incremented inside send_reset
            self._inference_ctx.send_reset()
        else:
            epoch = 0
        
        if self._debug:
            logger.info(
                f"[ActionManager] RESET | epoch→{epoch} | "
                f"stats: chunks_received={self._total_chunks_received}, "
                f"triggers_sent={self._total_triggers_sent}, "
                f"waits={self._total_waits}, "
                f"avg_wait={self._cumulative_wait_time / max(1, self._total_waits):.2f}ms"
            )
            # Reset per-rollout stats
            self._total_chunks_received = 0
            self._total_chunks_discarded = 0
            self._total_triggers_sent = 0
            self._total_waits = 0
            self._cumulative_wait_time = 0.0
