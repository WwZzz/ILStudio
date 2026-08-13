"""Transport-independent action chunk management."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value not in ("", "0", "false", "no", "off")


class AbstractActionChunkManager(ABC):
    """Interface for storing and dispatching policy-produced action chunks."""

    @abstractmethod
    def put(self, chunk, timestamp: float = None):
        """Store a newly produced action chunk."""

    @abstractmethod
    def get(self, timestamp: float = None):
        """Return one action step from the current chunk."""

    @abstractmethod
    def is_empty(self) -> bool:
        """Return whether no action step is available for dispatch."""

    @abstractmethod
    def should_infer(self) -> bool:
        """Return whether another chunk should be requested proactively."""

    @abstractmethod
    def reset(self):
        """Reset local chunk state."""


class BasicActionChunkManager(AbstractActionChunkManager):
    """Replace the current chunk when a new chunk arrives.

    This class is deliberately independent from SHM and policy inference. Its
    protected state remains stable because the existing action manager variants
    extend this behavior by overriding ``put`` or ``should_infer``.
    """

    def __init__(self, debug: bool = False, chunk_size: int = -1, **kwargs):
        self._lock = threading.Lock()
        self.t = 0
        self.current_step = 0
        self._buffer = None
        self._chunk_buffer = None
        self.chunk_size = -1
        self.set_chunk_size(chunk_size)
        self._debug = bool(debug or _env_flag("ILSTUDIO_ACTION_DEBUG"))
        self._stats = {
            "chunks_received": 0,
            "chunks_discarded": 0,
            "triggers_sent": 0,
            "waits": 0,
            "cumulative_wait_ms": 0.0,
            "total_remaining_on_replace": 0,
            "replace_count": 0,
        }

        if self._debug:
            logger.trace(
                f"[ActionManager] {self.__class__.__name__} initialized "
                "(debug=True)"
            )

    def _log_debug(self, message: str, *args):
        if self._debug:
            logger.trace(
                f"[ActionManager:{self.__class__.__name__}] " + message, *args
            )

    def _buffer_status(self):
        with self._lock:
            chunk_len = (
                len(self._chunk_buffer) if self._chunk_buffer is not None else 0
            )
            return self.current_step, chunk_len

    def _summarize_action(self, action) -> str:
        if action is None:
            return "action=None"
        arr = action
        if (
            isinstance(action, np.ndarray)
            and action.dtype == object
            and len(action) > 0
        ):
            first = action[0]
            if isinstance(first, dict):
                arr = first.get("action", None)
        if arr is None:
            return "action=None"
        arr = np.asarray(arr)
        if arr.size == 0:
            return f"shape={arr.shape}, empty"
        flat = arr.reshape(-1)
        preview = ", ".join(f"{float(x):.4f}" for x in flat[:4])
        return f"shape={arr.shape}, preview=[{preview}]"

    def set_chunk_size(self, chunk_size: int = -1) -> None:
        """Set the maximum number of actions accepted from each policy chunk.

        Positive values preserve the legacy ``MetaPolicy(chunk_size=...)``
        behavior. Zero, negative values, and ``None`` keep the full chunk.
        """
        if chunk_size is None:
            chunk_size = -1
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer or None")
        self.chunk_size = chunk_size

    def prepare_chunk(self, chunk):
        """Apply transport-independent limits before subclass chunk logic."""
        if chunk is None or self.chunk_size <= 0:
            return chunk
        return chunk[: self.chunk_size]

    def put(self, chunk, timestamp: float = None):
        chunk = self.prepare_chunk(chunk)
        with self._lock:
            old_len = len(self._chunk_buffer) if self._chunk_buffer is not None else 0
            if self._chunk_buffer is not None:
                remained = max(0, old_len - self.current_step)
                if remained > 0:
                    self._stats["total_remaining_on_replace"] += remained
                    self._stats["replace_count"] += 1
            self._chunk_buffer = chunk
            self.current_step = 0
        self._log_debug(
            "stored chunk_len={} replaced_old_len={}",
            len(chunk) if chunk is not None else 0,
            old_len,
        )

    def get(self, timestamp: float = None):
        with self._lock:
            if (
                self._chunk_buffer is None
                or self.current_step >= len(self._chunk_buffer)
            ):
                return None
            action = self._chunk_buffer[self.current_step]
            self.current_step += 1
            return action

    def is_empty(self) -> bool:
        with self._lock:
            if self._chunk_buffer is None:
                return True
            return self.current_step >= len(self._chunk_buffer)

    def should_infer(self) -> bool:
        """Request inference only when the executor observes an empty buffer."""
        return False

    def _reset_buffer_state(self) -> None:
        with self._lock:
            self._chunk_buffer = None
            self._buffer = None
            self.current_step = 0
            self.t = 0

    def _report_and_reset_stats(self) -> None:
        if self._debug:
            stats = self._stats
            avg_wait = stats["cumulative_wait_ms"] / max(1, stats["waits"])
            avg_wasted = (
                stats["total_remaining_on_replace"]
                / max(1, stats["replace_count"])
                if stats["replace_count"] > 0
                else 0.0
            )
            logger.info(
                f"[ActionManager] RESET | chunks: "
                f"recv={stats['chunks_received']} "
                f"discard={stats['chunks_discarded']} | "
                f"triggers={stats['triggers_sent']} waits={stats['waits']} "
                f"avg_wait={avg_wait:.1f}ms | "
                f"avg_unused_actions={avg_wasted:.1f}"
            )

        for key in self._stats:
            self._stats[key] = (
                0 if isinstance(self._stats[key], int) else 0.0
            )

    def reset(self):
        self._reset_buffer_state()
        self._report_and_reset_stats()
