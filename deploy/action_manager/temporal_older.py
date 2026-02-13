from loguru import logger
from .temporal_agg import TemporalAggManager


class TemporalOlderManager(TemporalAggManager):
    """Refuse newly coming chunks until the last chunk ends x%"""
    
    def __init__(self, coef: float = 0.1, older_coef: float = 0.75, **kwargs):
        super().__init__(coef=coef, **kwargs)
        self.older_coef = older_coef
        if self._debug:
            logger.info(f"[TemporalOlderManager] older_coef={self.older_coef}")

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
                if self._debug:
                    logger.debug(
                        f"[TemporalOlderManager] t={self.t} | Accepted first chunk: len={len(chunk)}"
                    )
        else:
            with self._lock:
                threshold = int(len(self._chunk_buffer) * self.older_coef)
                if self.current_step < threshold:
                    self._total_chunks_discarded += 1
                    if self._debug:
                        logger.debug(
                            f"[TemporalOlderManager] t={self.t} | REFUSED chunk "
                            f"(step {self.current_step}/{len(self._chunk_buffer)}, "
                            f"need ≥{threshold} = {self.older_coef*100:.0f}%) | "
                            f"discarded_total={self._total_chunks_discarded}"
                        )
                    return
            super().put(chunk, timestamp)
