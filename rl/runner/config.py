"""Validated configuration for the outer reinforcement-learning loop."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RLRunnerConfig:
    iterations: int = 1
    collect_steps: Optional[int] = None
    collect_episodes: Optional[int] = None
    batch_size: int = 1
    updates_per_iteration: int = 1
    warmup_steps: int = 0
    deterministic_collection: bool = False
    clear_rollout_after_update: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if (self.collect_steps is None) == (self.collect_episodes is None):
            raise ValueError("provide exactly one of collect_steps or collect_episodes")
        for name in ("collect_steps", "collect_episodes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.updates_per_iteration < 0:
            raise ValueError("updates_per_iteration cannot be negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
