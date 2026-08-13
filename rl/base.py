"""Shared data contracts for ILStudio reinforcement learning.

These contracts intentionally reuse :class:`benchmark.base.MetaObs` and
:class:`benchmark.base.MetaAction`.  RL components must not introduce a second
observation or action representation at the environment boundary.
"""

from dataclasses import dataclass, field
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import numpy as np

from benchmark.base import MetaAction, MetaObs


ACTION_POLICY_INFO_KEY = "action_policy_info"
RL_BOOTSTRAP_MASK_KEY = "rl.bootstrap_mask"
RL_TERMINATED_ON_SUCCESS_KEY = "rl.terminated_on_success"
RL_LIKELIHOOD_GROUP_KEY = "rl.likelihood_group"
RL_LIKELIHOOD_GROUP_SIZE_KEY = "rl.likelihood_group_size"


@dataclass
class PolicyOutput:
    """One policy decision and the metadata required to train from it.

    ``MetaAction.action`` may contain either one action or an action chunk.
    Algorithm-specific values such as log probabilities, values, token masks,
    and policy versions live in ``policy_info`` so the ILStudio action format
    remains unchanged.

    A chunk policy may put sequence-valued, step-specific metadata under
    ``ACTION_POLICY_INFO_KEY``. ``RLPolicyExecutor`` removes that container and
    merges the corresponding entry into each step-level output.
    """

    action: MetaAction
    policy_info: Dict[str, Any] = field(default_factory=dict)
    decision: Optional["PolicyDecision"] = None
    action_offset: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, MetaAction):
            raise TypeError("action must be MetaAction")
        if not isinstance(self.policy_info, dict):
            raise TypeError("policy_info must be a dict")
        if self.decision is not None and not isinstance(self.decision, PolicyDecision):
            raise TypeError("decision must be PolicyDecision or None")
        if self.action_offset is not None and (
            isinstance(self.action_offset, bool)
            or not isinstance(self.action_offset, int)
            or self.action_offset < 0
        ):
            raise ValueError("action_offset must be a non-negative integer or None")
        if self.decision is None and self.action_offset is not None:
            raise ValueError("action_offset requires a policy decision")
        if (
            self.decision is not None
            and self.action_offset is not None
            and self.action_offset >= self.decision.chunk_size
        ):
            raise ValueError("action_offset exceeds policy decision chunk")
        self.policy_info = dict(self.policy_info)


@dataclass
class MetaTransition:
    """A single ILStudio environment transition with RL-specific metadata."""

    obs: MetaObs
    action: MetaAction
    next_obs: MetaObs
    reward: Dict[str, Any]
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    policy_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.obs, MetaObs):
            raise TypeError("obs must be MetaObs")
        if not isinstance(self.action, MetaAction):
            raise TypeError("action must be MetaAction")
        if not isinstance(self.next_obs, MetaObs):
            raise TypeError("next_obs must be MetaObs")
        if not isinstance(self.reward, dict):
            raise TypeError("reward must be a dict")
        if not all(isinstance(key, str) for key in self.reward):
            raise TypeError("reward keys must be strings")
        if not isinstance(self.info, dict):
            raise TypeError("info must be a dict")
        if not isinstance(self.policy_info, dict):
            raise TypeError("policy_info must be a dict")
        if not isinstance(self.terminated, (bool, np.bool_)):
            raise TypeError("terminated must be a bool")
        if not isinstance(self.truncated, (bool, np.bool_)):
            raise TypeError("truncated must be a bool")
        if RL_BOOTSTRAP_MASK_KEY in self.info and not isinstance(
            self.info[RL_BOOTSTRAP_MASK_KEY], (bool, np.bool_)
        ):
            raise TypeError(f"{RL_BOOTSTRAP_MASK_KEY} must be a bool")

        # Detach mutable containers supplied by envs, reward modules, or
        # executors. Array/tensor values are deliberately not copied here.
        self.reward = dict(self.reward)
        self.info = dict(self.info)
        self.policy_info = dict(self.policy_info)
        self.terminated = bool(self.terminated)
        self.truncated = bool(self.truncated)

    @property
    def episode_done(self) -> bool:
        """Whether the environment requires reset after this transition."""

        return self.terminated or self.truncated

    @property
    def bootstrap_mask(self) -> bool:
        """Whether value targets may bootstrap from ``next_obs``.

        Gymnasium time-limit truncation permits bootstrapping.  A true MDP
        termination does not.
        """

        configured = self.info.get(RL_BOOTSTRAP_MASK_KEY)
        if configured is None:
            return not self.terminated
        return not self.terminated and bool(configured)

    @property
    def success(self) -> bool:
        """Task success, kept independent from episode termination."""

        return bool(self.info.get("success", False))


# ``Transition`` is the canonical name for new RL code. ``MetaTransition`` is
# retained because the first ILStudio RL modules exposed it publicly.
Transition = MetaTransition


def _shape(value):
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(value).shape
        except Exception:
            return None
    return tuple(int(item) for item in shape)


@dataclass(frozen=True)
class PolicyTrace:
    """Likelihood metadata for one policy decision.

    ``axis_names`` gives semantic names to the axes in ``old_logprobs``.  For
    example, chunk policies normally use ``("action",)``, diffusion policies
    may use ``("denoise", "action")``, and autoregressive VLA policies use
    ``("token",)``.  Algorithms can therefore assign credit without knowing
    how a policy produced its action.
    """

    kind: str
    old_logprobs: Any
    valid_mask: Any = None
    axis_names: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise TypeError("policy trace kind must be a non-empty string")
        axis_names = tuple(self.axis_names)
        if not all(isinstance(name, str) and name for name in axis_names):
            raise TypeError("policy trace axis names must be non-empty strings")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("policy trace axis names must be unique")
        trace_shape = _shape(self.old_logprobs)
        if trace_shape is not None and axis_names and len(trace_shape) != len(axis_names):
            raise ValueError("policy trace axis names must match log-probability rank")
        mask_shape = _shape(self.valid_mask)
        if mask_shape is not None and trace_shape is not None and mask_shape != trace_shape:
            raise ValueError("policy trace valid mask must match log-probability shape")
        object.__setattr__(self, "axis_names", axis_names)
        object.__setattr__(self, "extras", dict(self.extras))


@dataclass(frozen=True)
class PolicyDecision:
    """One policy inference, whose action may be an action chunk."""

    decision_id: Hashable
    obs: MetaObs
    action: MetaAction
    trace: Optional[PolicyTrace] = None
    value: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decision_id is None or not isinstance(self.decision_id, Hashable):
            raise TypeError("decision_id must be a non-None hashable value")
        if not isinstance(self.obs, MetaObs):
            raise TypeError("policy decision observation must be MetaObs")
        if not isinstance(self.action, MetaAction):
            raise TypeError("policy decision action must be MetaAction")
        if self.trace is not None and not isinstance(self.trace, PolicyTrace):
            raise TypeError("policy decision trace must be PolicyTrace or None")
        if len(self.action) <= 0:
            raise ValueError("policy decision action cannot be empty")
        object.__setattr__(self, "extras", dict(self.extras))

    @property
    def chunk_size(self) -> int:
        return len(self.action)


@dataclass(frozen=True)
class RolloutStep:
    """One atomic environment transition linked to its policy decision."""

    transition: Transition
    decision_id: Hashable
    action_offset: int = 0

    def __post_init__(self):
        if not isinstance(self.transition, MetaTransition):
            raise TypeError("rollout step transition must be Transition")
        if self.decision_id is None or not isinstance(self.decision_id, Hashable):
            raise TypeError("rollout step decision_id must be hashable")
        if isinstance(self.action_offset, bool) or not isinstance(self.action_offset, int):
            raise TypeError("rollout step action_offset must be an integer")
        if self.action_offset < 0:
            raise ValueError("rollout step action_offset cannot be negative")


@dataclass(frozen=True)
class DecisionTransition:
    """Logical chunk-level view over consecutive atomic environment steps."""

    decision: PolicyDecision
    steps: Tuple[RolloutStep, ...]

    def __post_init__(self):
        if not isinstance(self.decision, PolicyDecision):
            raise TypeError("decision transition requires PolicyDecision")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("decision transition must contain at least one step")
        offsets = []
        for step in steps:
            if not isinstance(step, RolloutStep):
                raise TypeError("decision transition steps must be RolloutStep")
            if step.decision_id != self.decision.decision_id:
                raise ValueError("rollout step refers to a different policy decision")
            if step.action_offset >= self.decision.chunk_size:
                raise ValueError("rollout step action_offset exceeds decision chunk")
            offsets.append(step.action_offset)
        if offsets != list(range(offsets[0], offsets[0] + len(offsets))):
            raise ValueError("decision transition action offsets must be consecutive")
        object.__setattr__(self, "steps", steps)

    @property
    def transitions(self) -> Tuple[Transition, ...]:
        return tuple(step.transition for step in self.steps)

    @property
    def terminated(self) -> bool:
        return any(step.transition.terminated for step in self.steps)

    @property
    def truncated(self) -> bool:
        return any(step.transition.truncated for step in self.steps)

    @property
    def episode_done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def duration(self) -> int:
        """Number of atomic environment steps executed for this decision."""

        return len(self.steps)

    @property
    def obs(self) -> MetaObs:
        """Observation at which the action chunk was produced."""

        return self.decision.obs

    @property
    def next_obs(self) -> MetaObs:
        """Observation after the last executed action in the chunk."""

        return self.steps[-1].transition.next_obs

    @property
    def action(self) -> MetaAction:
        """Full action emitted by the policy decision."""

        return self.decision.action

    @property
    def bootstrap_mask(self) -> bool:
        """Whether an SMDP target may bootstrap after this decision."""

        return self.transitions[-1].bootstrap_mask

    @property
    def success(self) -> bool:
        return any(step.transition.success for step in self.steps)

    def reward_sum(self, key: str) -> float:
        if not isinstance(key, str) or not key:
            raise TypeError("reward key must be a non-empty string")
        total = 0.0
        for transition in self.transitions:
            if key not in transition.reward:
                raise KeyError(f"transition reward is missing key {key!r}")
            value = np.asarray(transition.reward[key])
            if value.size != 1:
                raise TypeError(f"reward {key!r} must be scalar per env step")
            total += float(value.reshape(-1)[0])
        return total

    def discounted_reward(self, key: str, gamma: float) -> float:
        """Discount atomic rewards within this variable-duration decision."""

        if not isinstance(key, str) or not key:
            raise TypeError("reward key must be a non-empty string")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise TypeError("gamma must be a real number")
        gamma = float(gamma)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        total = 0.0
        discount = 1.0
        for transition in self.transitions:
            if key not in transition.reward:
                raise KeyError(f"transition reward is missing key {key!r}")
            value = np.asarray(transition.reward[key])
            if value.size != 1:
                raise TypeError(f"reward {key!r} must be scalar per env step")
            scalar = float(value.reshape(-1)[0])
            if not np.isfinite(scalar):
                raise ValueError(f"reward {key!r} must be finite")
            total += discount * scalar
            discount *= gamma
        return total


@dataclass(frozen=True)
class Rollout:
    """Raw on-policy experience: atomic steps plus policy decisions."""

    steps: Tuple[RolloutStep, ...]
    decisions: Tuple[PolicyDecision, ...]

    def __post_init__(self):
        steps = tuple(self.steps)
        decisions = tuple(self.decisions)
        if not all(isinstance(step, RolloutStep) for step in steps):
            raise TypeError("rollout steps must be RolloutStep values")
        if not all(isinstance(decision, PolicyDecision) for decision in decisions):
            raise TypeError("rollout decisions must be PolicyDecision values")
        ids = [decision.decision_id for decision in decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("rollout decision ids must be unique")
        known_ids = set(ids)
        if any(step.decision_id not in known_ids for step in steps):
            raise ValueError("rollout step refers to an unknown policy decision")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "decisions", decisions)
        # Exercise the stricter per-decision invariants at construction time.
        self.decision_transitions()

    @property
    def transitions(self) -> Tuple[Transition, ...]:
        return tuple(step.transition for step in self.steps)

    def decision_transitions(self) -> Tuple[DecisionTransition, ...]:
        grouped = {decision.decision_id: [] for decision in self.decisions}
        for step in self.steps:
            grouped[step.decision_id].append(step)
        return tuple(
            DecisionTransition(decision=decision, steps=tuple(grouped[decision.decision_id]))
            for decision in self.decisions
            if grouped[decision.decision_id]
        )


@dataclass(frozen=True)
class CreditAssignment:
    """Advantages and their mapping onto a policy trace's objective axes."""

    advantages: Any
    objective_to_advantage: Any
    mask: Any = None

    def __post_init__(self):
        advantage_shape = _shape(self.advantages)
        mapping_shape = _shape(self.objective_to_advantage)
        if advantage_shape is None or len(advantage_shape) != 1:
            raise ValueError("credit advantages must be one-dimensional")
        if mapping_shape is None:
            raise ValueError("credit objective mapping must be array-like")
        mask_shape = _shape(self.mask)
        if mask_shape is not None and mask_shape != mapping_shape:
            raise ValueError("credit mask must match objective mapping shape")


@dataclass(frozen=True)
class PolicyObjective:
    """Algorithm-ready old/new likelihoods after credit assignment."""

    old_logprobs: Any
    new_logprobs: Any
    advantages: Any
    mask: Any = None
    values: Any = None
    returns: Any = None
    axis_names: Tuple[str, ...] = ()

    def __post_init__(self):
        old_shape = _shape(self.old_logprobs)
        new_shape = _shape(self.new_logprobs)
        advantage_shape = _shape(self.advantages)
        if old_shape is None or old_shape != new_shape:
            raise ValueError("old and new log probabilities must have equal shape")
        if advantage_shape != old_shape:
            raise ValueError("objective advantages must match log-probability shape")
        mask_shape = _shape(self.mask)
        if mask_shape is not None and mask_shape != old_shape:
            raise ValueError("objective mask must match log-probability shape")
        axis_names = tuple(self.axis_names)
        if axis_names and len(axis_names) != len(old_shape):
            raise ValueError("objective axis names must match objective rank")
        object.__setattr__(self, "axis_names", axis_names)

    @property
    def ratio(self):
        difference = self.new_logprobs - self.old_logprobs
        try:
            import torch

            if torch.is_tensor(difference):
                return torch.exp(difference)
        except ImportError:
            pass
        return np.exp(difference)
