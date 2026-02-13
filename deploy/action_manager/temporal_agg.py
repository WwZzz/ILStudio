from loguru import logger
from .base import BasicActionManager


class TemporalAggManager(BasicActionManager):
    """Exponentially average the last and the new chunks for better smoothness"""
    
    def __init__(self, coef: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        if self._debug:
            logger.info(f"[TemporalAggManager] coef={self.coef}")

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            with self._lock:
                prev_step = self.current_step
                prev_len = len(self._chunk_buffer)
                remain_len = prev_len - prev_step
                blend_count = min(remain_len, len(chunk)) if remain_len > 0 else 0
                if remain_len > 0:
                    for idx in range(remain_len):
                        chunk[idx]['action'] = (1. - self.coef) * chunk[idx]['action'] + self.coef * self._chunk_buffer[idx + prev_step]['action']
                self._chunk_buffer = chunk
                self.current_step = 0
                if self._debug:
                    logger.debug(
                        f"[TemporalAggManager] t={self.t} | PUT: blended {blend_count} steps "
                        f"(old {prev_step}/{prev_len}, remain={remain_len}) | "
                        f"new_chunk_len={len(chunk)} | coef={self.coef}"
                    )
