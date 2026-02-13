import time
from loguru import logger
from .base import BasicActionManager


class DelayFreeManager(BasicActionManager):
    """Remove the outdated actions from each chunk"""
    
    def __init__(self, duration: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.duration = duration
        if self._debug:
            logger.info(f"[DelayFreeManager] duration={self.duration}s")

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            delay_time = time.perf_counter() - timestamp
            delayed_start_idx = int(delay_time // self.duration)
            original_len = len(chunk)

            if delayed_start_idx < len(chunk):
                chunk = chunk[delayed_start_idx:]
                with self._lock:
                    self._chunk_buffer = chunk
                    self.current_step = 0
                    if self._debug:
                        logger.debug(
                            f"[DelayFreeManager] t={self.t} | PUT: delay={delay_time*1000:.1f}ms | "
                            f"skipped {delayed_start_idx}/{original_len} steps | "
                            f"effective_len={len(chunk)}"
                        )
            else:
                self._total_chunks_discarded += 1
                if self._debug:
                    logger.debug(
                        f"[DelayFreeManager] t={self.t} | DISCARDED chunk: delay={delay_time*1000:.1f}ms | "
                        f"skip_idx={delayed_start_idx} ≥ chunk_len={original_len}"
                    )
