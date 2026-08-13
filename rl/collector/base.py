"""Collector results shared by RL interaction implementations."""

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Real
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from benchmark.base import MetaObs
from rl.base import (
    RL_BOOTSTRAP_MASK_KEY,
    RL_TERMINATED_ON_SUCCESS_KEY,
    MetaTransition,
    Rollout,
)


class _CollectorRuntime:
    """Accumulate collection timings and utilization without changing APIs."""

    _PHASES = ("reset", "policy", "env", "reward")

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.started_at = perf_counter()
        self.phase_seconds = {phase: 0.0 for phase in self._PHASES}
        self.active_env_counts = []
        self.inference_batch_sizes = []

    @contextmanager
    def measure(self, phase: str):
        if phase not in self.phase_seconds:
            raise KeyError(f"unknown collector phase: {phase}")
        started_at = perf_counter()
        try:
            yield
        finally:
            self.phase_seconds[phase] += perf_counter() - started_at

    def record_step(self, *, active_envs: int, inference_batch_size: int) -> None:
        self.active_env_counts.append(int(active_envs))
        if inference_batch_size > 0:
            self.inference_batch_sizes.append(int(inference_batch_size))

    @staticmethod
    def _summary(values, prefix):
        if not values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_mean": sum(values) / len(values),
            f"{prefix}_min": min(values),
            f"{prefix}_max": max(values),
        }

    def finish(self) -> Dict[str, float]:
        total_seconds = perf_counter() - self.started_at
        accounted_seconds = sum(self.phase_seconds.values())
        metrics = {
            "runtime/collector_seconds": total_seconds,
            "runtime/collector_reset_seconds": self.phase_seconds["reset"],
            "runtime/collector_policy_seconds": self.phase_seconds["policy"],
            "runtime/collector_env_seconds": self.phase_seconds["env"],
            "runtime/collector_reward_seconds": self.phase_seconds["reward"],
            "runtime/collector_overhead_seconds": max(
                0.0, total_seconds - accounted_seconds
            ),
            "collector/step_batches": len(self.active_env_counts),
            "collector/inference_calls": len(self.inference_batch_sizes),
        }
        metrics.update(
            self._summary(self.active_env_counts, "collector/active_envs")
        )
        metrics.update(
            self._summary(
                self.inference_batch_sizes,
                "collector/inference_batch_size",
            )
        )
        active_mean = metrics["collector/active_envs_mean"]
        metrics["collector/env_capacity_utilization"] = (
            active_mean / self.num_envs if self.num_envs > 0 else 0.0
        )
        return metrics


def _inference_batch_size(outputs) -> int:
    """Infer how many outputs came from a new policy decision this step."""

    outputs = tuple(outputs)
    if not outputs:
        return 0
    if all(output.decision is None for output in outputs):
        return len(outputs)
    return sum(
        output.decision is not None and output.action_offset in (None, 0)
        for output in outputs
    )


def annotate_episode_timestep(obs: MetaObs, timestep: int) -> MetaObs:
    """Attach RL-local episode progress without mutating benchmark observations."""

    if not isinstance(obs, MetaObs):
        raise TypeError("RL collectors require MetaObs observations")
    if (
        isinstance(timestep, bool)
        or not isinstance(timestep, int)
        or timestep < 0
    ):
        raise ValueError("episode timestep must be a non-negative integer")
    return replace(obs, timestep=timestep)


def validate_episode_semantics(
    *,
    terminate_on_success: bool,
    bootstrap_on_truncation: bool,
):
    values = {
        "terminate_on_success": terminate_on_success,
        "bootstrap_on_truncation": bootstrap_on_truncation,
    }
    for name, value in values.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")
    return terminate_on_success, bootstrap_on_truncation


def validate_max_episode_steps(max_episode_steps):
    if max_episode_steps is None:
        return None
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise ValueError("max_episode_steps must be a positive integer or None")
    return max_episode_steps


def apply_episode_time_limit(
    transition: MetaTransition,
    *,
    episode_step: int,
    max_episode_steps,
) -> MetaTransition:
    """Truncate an active episode at an RL-local step limit."""

    if not isinstance(transition, MetaTransition):
        raise TypeError("episode time limits require MetaTransition")
    if (
        isinstance(episode_step, bool)
        or not isinstance(episode_step, int)
        or episode_step <= 0
    ):
        raise ValueError("episode_step must be a positive integer")
    max_episode_steps = validate_max_episode_steps(max_episode_steps)
    if (
        max_episode_steps is None
        or episode_step < max_episode_steps
        or transition.episode_done
    ):
        return transition
    info = dict(transition.info)
    info["rl.time_limit_reached"] = True
    info["rl.time_limit_steps"] = max_episode_steps
    return replace(transition, truncated=True, info=info)


def apply_episode_semantics(
    transition: MetaTransition,
    *,
    terminate_on_success: bool,
    bootstrap_on_truncation: bool,
) -> MetaTransition:
    """Apply task-level episode semantics without changing the benchmark API."""

    if not isinstance(transition, MetaTransition):
        raise TypeError("episode semantics require MetaTransition")
    terminate_on_success, bootstrap_on_truncation = validate_episode_semantics(
        terminate_on_success=terminate_on_success,
        bootstrap_on_truncation=bootstrap_on_truncation,
    )
    info = dict(transition.info)
    terminated = transition.terminated
    truncated = transition.truncated
    if terminate_on_success and transition.success and not terminated:
        info["rl.original_terminated"] = terminated
        info["rl.original_truncated"] = truncated
        info[RL_TERMINATED_ON_SUCCESS_KEY] = True
        terminated = True
        truncated = False
    if terminated:
        info[RL_BOOTSTRAP_MASK_KEY] = False
    elif truncated:
        info[RL_BOOTSTRAP_MASK_KEY] = bootstrap_on_truncation
    return replace(
        transition,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@dataclass(frozen=True)
class EpisodeSummary:
    index: int
    length: int
    reward: Dict[str, Any]
    terminated: bool
    truncated: bool
    success: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "reward", dict(self.reward))
        object.__setattr__(self, "info", dict(self.info))


@dataclass(frozen=True)
class CollectResult:
    transitions: Tuple[MetaTransition, ...]
    episodes: Tuple[EpisodeSummary, ...]
    rollout: Optional[Rollout] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "episodes", tuple(self.episodes))
        validated_metrics = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key:
                raise TypeError("collection metric keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"collection metric '{key}' must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"collection metric '{key}' must be finite")
            validated_metrics[key] = value
        object.__setattr__(self, "metrics", validated_metrics)
        if self.rollout is not None:
            if not isinstance(self.rollout, Rollout):
                raise TypeError("collection rollout must be Rollout or None")
            if self.rollout.transitions != self.transitions:
                raise ValueError("collection rollout must contain collected transitions")

    @property
    def num_steps(self) -> int:
        return len(self.transitions)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


class BaseCollector(ABC):
    @abstractmethod
    def collect(self, *, num_steps=None, num_episodes=None, **kwargs) -> CollectResult:
        """Collect an exact step or episode target."""

    @abstractmethod
    def close(self) -> None:
        """Release executor and environment resources."""

    def policy_updated(self) -> None:
        """Drop collector-side outputs cached under the previous policy."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
