"""Validated configuration for the outer reinforcement-learning loop."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RLRunnerConfig:
    mode: str = "online"
    iterations: int = 1
    collect_steps: Optional[int] = None
    collect_episodes: Optional[int] = None
    batch_size: int = 1
    updates_per_iteration: int = 1
    warmup_steps: int = 0
    deterministic_collection: bool = False
    clear_rollout_after_update: bool = True
    max_collection_attempts: int = 1
    offline_pretrain_iterations: int = 0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.mode not in {"online", "offline", "hybrid"}:
            raise ValueError("mode must be online, offline, or hybrid")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        has_steps = self.collect_steps is not None
        has_episodes = self.collect_episodes is not None
        if self.mode == "offline":
            if has_steps or has_episodes:
                raise ValueError("offline mode cannot configure environment collection")
        elif has_steps == has_episodes:
            raise ValueError("online and hybrid modes require one collection target")
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
        if (
            isinstance(self.max_collection_attempts, bool)
            or not isinstance(self.max_collection_attempts, int)
            or self.max_collection_attempts <= 0
        ):
            raise ValueError("max_collection_attempts must be positive")
        if (
            isinstance(self.offline_pretrain_iterations, bool)
            or not isinstance(self.offline_pretrain_iterations, int)
            or self.offline_pretrain_iterations < 0
        ):
            raise ValueError("offline_pretrain_iterations cannot be negative")
        if self.mode != "hybrid" and self.offline_pretrain_iterations:
            raise ValueError("offline_pretrain_iterations requires hybrid mode")
