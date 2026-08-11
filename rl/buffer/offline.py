"""Read-only replay buffer backed by an offline task dataset."""

from __future__ import annotations

import copy

import numpy as np

from data_utils.offline_rl import OfflineReplayDataset

from .base import BaseBuffer, BufferBatch, TransitionStorage


class OfflineReplayBuffer(BaseBuffer):
    """Read-only replay buffer that samples task data lazily."""

    STATE_VERSION = 1

    def __init__(self, dataset: OfflineReplayDataset, *, seed=None):
        if not isinstance(dataset, OfflineReplayDataset):
            raise TypeError("offline replay buffer requires OfflineReplayDataset")
        # BaseBuffer owns common type checks, but offline items remain in the
        # source dataset instead of being duplicated in an in-memory storage.
        super().__init__(TransitionStorage())
        self.dataset = dataset
        self._rng = np.random.default_rng(seed)

    @property
    def buffer_type(self):
        return "replay"

    @property
    def capacity(self):
        return len(self.dataset)

    @property
    def transitions(self):
        return tuple(self.dataset[index] for index in range(len(self)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def add(self, transition):
        del transition
        raise RuntimeError("offline replay data is read-only")

    def extend(self, transitions):
        del transitions
        raise RuntimeError("offline replay data is read-only")

    def clear(self):
        # Rejecting an online collection must never erase an offline dataset.
        return None

    def get_batch(self, indices):
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        return BufferBatch(
            transitions=tuple(self.dataset[int(index)] for index in indices),
            indices=indices,
            metadata={
                "offline_samples": len(indices),
                "online_samples": 0,
            },
        )

    def sample(self, batch_size, *, replace=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not replace and batch_size > len(self):
            raise ValueError("batch_size exceeds offline replay size")
        indices = self._rng.choice(len(self), size=batch_size, replace=replace)
        return self.get_batch(indices)

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "buffer_type": self.buffer_type,
            "dataset_length": len(self),
            "item_type": self.dataset.item_type,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported offline replay state")
        if state.get("buffer_type") != self.buffer_type:
            raise ValueError("offline replay buffer type mismatch")
        if state.get("dataset_length") != len(self):
            raise ValueError("offline replay dataset length mismatch")
        if state.get("item_type") != self.dataset.item_type:
            raise ValueError("offline replay item type mismatch")
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
__all__ = ["OfflineReplayBuffer"]
