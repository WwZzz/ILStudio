"""Collector results shared by RL interaction implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from benchmark.base import MetaObs
from rl.base import (
    RL_BOOTSTRAP_MASK_KEY,
    RL_TERMINATED_ON_SUCCESS_KEY,
    MetaTransition,
    Rollout,
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

    def __post_init__(self):
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "episodes", tuple(self.episodes))
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
