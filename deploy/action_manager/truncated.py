from loguru import logger
from .base import BasicActionManager


class TruncatedManager(BasicActionManager):
    """
    Truncate action chunks by dropping the first and last portions.
    
    This manager discards:
    - First len(chunk) * start_ratio actions (warm-up phase) - NOT applied to the first chunk
    - Last len(chunk) * end_ratio actions (uncertain predictions)
    
    Special behavior: The first chunk keeps all beginning actions (start_ratio is ignored)
    to ensure smooth startup. Subsequent chunks apply both start_ratio and end_ratio.
    
    Then executes the remaining middle portion like OlderFirstManager,
    refusing new chunks until the current chunk is sufficiently executed.
    """
    
    def __init__(self, start_ratio: float = 0.0, end_ratio: float = 0.0, older_coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.older_coef = older_coef
        
        if self._debug:
            logger.info(
                f"[TruncatedManager] start_ratio={self.start_ratio} "
                f"end_ratio={self.end_ratio} older_coef={self.older_coef}"
            )

    def put(self, chunk, timestamp: float = None):
        # Truncate the chunk first
        if chunk is not None and len(chunk) > 0:
            total_len = len(chunk)
            
            # For the first chunk, don't drop the beginning (start_idx = 0)
            # For subsequent chunks, apply start_ratio normally
            is_first_chunk = (self._chunk_buffer is None)
            start_idx = 0 if is_first_chunk else int(total_len * self.start_ratio)
            end_idx = total_len - int(total_len * self.end_ratio)
            
            # Ensure we have at least one action
            if end_idx <= start_idx:
                # If truncation would remove everything, keep at least the middle action
                mid_idx = total_len // 2
                chunk = [chunk[mid_idx]]
                if self._debug:
                    logger.warning(
                        f"[TruncatedManager] t={self.t} | Truncation too aggressive, "
                        f"keeping only middle action (original_len={total_len})"
                    )
            else:
                chunk = chunk[start_idx:end_idx]
                if self._debug and (start_idx > 0 or end_idx < total_len):
                    chunk_type = "first" if is_first_chunk else "subsequent"
                    logger.debug(
                        f"[TruncatedManager] t={self.t} | Truncated ({chunk_type}): "
                        f"{total_len}→{len(chunk)} [{start_idx}:{end_idx}]"
                    )
        
        # Now apply OlderFirst logic
        if self._chunk_buffer is None:
            # First chunk, accept directly
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
                if self._debug:
                    logger.debug(
                        f"[TruncatedManager] t={self.t} | Accepted first chunk: len={len(chunk)}"
                    )
        else:
            # Check if current chunk is sufficiently executed
            with self._lock:
                threshold = int(len(self._chunk_buffer) * self.older_coef)
                if self.current_step < threshold:
                    # Current chunk not done enough, refuse new chunk
                    self._total_chunks_discarded += 1
                    if self._debug:
                        logger.debug(
                            f"[TruncatedManager] t={self.t} | REFUSED chunk: "
                            f"step {self.current_step}/{len(self._chunk_buffer)} "
                            f"(need ≥{threshold} = {self.older_coef*100:.0f}%) | "
                            f"discarded_total={self._total_chunks_discarded}"
                        )
                    return
            
            # Accept new chunk
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
                if self._debug:
                    logger.debug(
                        f"[TruncatedManager] t={self.t} | Accepted new chunk: len={len(chunk)}"
                    )
