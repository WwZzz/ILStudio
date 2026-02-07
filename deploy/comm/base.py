#!/usr/bin/env python3
"""
Base classes for communication layer (Server / Client).

All concrete implementations (TCP pickle, FastAPI HTTP/JSON, etc.) should inherit from these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np


class BaseServer(ABC):
    """Abstract base class for policy servers."""

    @abstractmethod
    def start(self) -> None:
        """Start the server (blocking)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the server gracefully."""
        ...


class BaseClient(ABC):
    """
    Abstract base class for policy clients.

    Clients must implement `send_meta_obs` and provide a `select_action` helper
    that manages an internal action queue (chunk semantics).
    """

    @abstractmethod
    def send_meta_obs(self, meta_obs: Any, **kwargs) -> List[np.ndarray]:
        """
        Send a MetaObs to the server and return the list of action arrays.

        Returns:
            List[np.ndarray]: Each element is an object-dtype array where each
            item is a dict compatible with MetaAction.
        """
        ...

    @abstractmethod
    def select_action(self, mobs: Any, t: int, return_all: bool = False) -> Any:
        """
        Select action(s) for timestep `t`.

        Internally manages an action queue; requests new chunk from server when
        the queue is empty or when chunk boundary is reached.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g., clear action queue)."""
        ...

    def close(self) -> None:
        """Close any network connections. Default is no-op."""
        pass

    # -------------------------------------------------------------------------
    # Optional properties for compatibility with existing code
    # -------------------------------------------------------------------------
    @property
    def ctrl_space(self) -> str:
        return getattr(self, "_ctrl_space", "ee")

    @ctrl_space.setter
    def ctrl_space(self, value: str) -> None:
        self._ctrl_space = value

    @property
    def ctrl_type(self) -> str:
        return getattr(self, "_ctrl_type", "delta")

    @ctrl_type.setter
    def ctrl_type(self, value: str) -> None:
        self._ctrl_type = value

    @property
    def chunk_size(self) -> Optional[int]:
        return getattr(self, "_chunk_size", None)

    @chunk_size.setter
    def chunk_size(self, value: Optional[int]) -> None:
        self._chunk_size = value

