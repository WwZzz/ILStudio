from loguru import logger
from .base import BasicActionManager


class OlderFirstManager(BasicActionManager):
    """Refuse newly coming chunks unless the last chunk ends x%"""
    
    def __init__(self, coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        if self._debug:
            logger.info(f"[OlderFirstManager] coef={self.coef}")

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            super().put(chunk, timestamp)
        else:
            with self._lock:
                threshold = int(len(self._chunk_buffer) * self.coef)
                if self.current_step < threshold:
                    self._total_chunks_discarded += 1
                    if self._debug:
                        logger.debug(
                            f"[OlderFirstManager] t={self.t} | REFUSED chunk "
                            f"(step {self.current_step}/{len(self._chunk_buffer)}, "
                            f"need ≥{threshold} = {self.coef*100:.0f}%) | "
                            f"discarded_total={self._total_chunks_discarded}"
                        )
                    return
            super().put(chunk, timestamp)
