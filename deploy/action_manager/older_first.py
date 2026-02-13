from .base import BasicActionManager


class OlderFirstManager(BasicActionManager):
    """Refuse newly coming chunks unless the last chunk ends x%"""
    
    def __init__(self, coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            super().put(chunk, timestamp)
        else:
            with self._lock:
                threshold = int(len(self._chunk_buffer) * self.coef)
                if self.current_step < threshold:
                    self._stats["chunks_discarded"] += 1
                    return
            super().put(chunk, timestamp)
