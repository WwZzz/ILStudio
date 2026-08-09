"""Critic interfaces shared by independent and policy-hook implementations."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn


class BaseCritic(nn.Module, ABC):
    """Map ILStudio observations to one scalar value per observation."""

    STATE_VERSION = 1
    STATE_FILENAME = "critic.pt"

    @abstractmethod
    def forward(self, obs, *, context=None):
        pass

    @classmethod
    def _critic_type(cls):
        return f"{cls.__module__}.{cls.__qualname__}"

    @staticmethod
    def _checkpoint_root(checkpoint):
        root = Path(checkpoint)
        if root.name.startswith("checkpoint-"):
            root = root.parent
        return root

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.STATE_FILENAME
        torch.save(
            {
                "version": self.STATE_VERSION,
                "critic_type": self._critic_type(),
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    def load_pretrained(self, checkpoint):
        path = self._checkpoint_root(checkpoint) / self.STATE_FILENAME
        if not path.is_file():
            return False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("critic checkpoint must contain a mapping")
        if payload.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported critic checkpoint version")
        critic_type = payload.get("critic_type")
        expected_type = self._critic_type()
        if critic_type != expected_type:
            raise TypeError(
                f"critic checkpoint contains {critic_type!r}, "
                f"expected {expected_type!r}"
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("critic checkpoint state_dict must be a mapping")
        self.load_state_dict(state_dict)
        return True
