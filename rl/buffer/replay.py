"""Off-policy replay buffer."""

import copy
from typing import Any, Dict

import numpy as np

from .base import BaseBuffer, BufferBatch, TransitionStorage
from rl.base import DecisionTransition, PolicyDecision, RolloutStep


class ReplayBuffer(BaseBuffer):
    """Bounded replay storage with oldest-first ring eviction."""

    def __init__(self, capacity: int, *, seed=None) -> None:
        super().__init__(TransitionStorage(capacity=capacity, overwrite=True))
        self._rng = np.random.default_rng(seed)

    @property
    def buffer_type(self) -> str:
        return "replay"

    def sample(self, batch_size: int, *, replace: bool = False) -> BufferBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self) == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if not replace and batch_size > len(self):
            raise ValueError(
                "batch_size exceeds replay size for sampling without replacement"
            )

        indices = self._rng.choice(
            len(self),
            size=batch_size,
            replace=replace,
        )
        return self.get_batch(indices)

    def _state_metadata(self) -> Dict[str, Any]:
        return {"rng_state": copy.deepcopy(self._rng.bit_generator.state)}

    def _validate_state_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        num_transitions: int,
    ) -> None:
        del num_transitions
        if not isinstance(metadata.get("rng_state"), dict):
            raise TypeError("replay rng_state metadata must be a dict")

    def _load_state_metadata(self, metadata: Dict[str, Any]) -> None:
        self._rng.bit_generator.state = copy.deepcopy(metadata["rng_state"])


class _DecisionTransitionStorage(TransitionStorage):
    @staticmethod
    def _validate_transition(transition) -> None:
        if not isinstance(transition, DecisionTransition):
            raise TypeError("decision replay entries must be DecisionTransition")


class DecisionReplayBuffer(ReplayBuffer):
    """Replay complete or terminally-shortened policy decisions."""

    def __init__(self, capacity: int, *, seed=None) -> None:
        BaseBuffer.__init__(
            self,
            _DecisionTransitionStorage(capacity=capacity, overwrite=True),
        )
        self._rng = np.random.default_rng(seed)
        self._pending = {}

    @property
    def num_env_steps(self) -> int:
        return sum(item.duration for item in self.storage) + sum(
            len(steps) for _, steps in self._pending.values()
        )

    def _validate_extend(self, transitions) -> None:
        for transition in transitions:
            _DecisionTransitionStorage._validate_transition(transition)

    def _flush_pending(self, decision_id):
        pending = self._pending.pop(decision_id, None)
        if pending is None:
            return None
        decision, steps = pending
        if not steps:
            raise RuntimeError("decision replay has a decision without steps")
        transition = DecisionTransition(
            decision=decision,
            steps=tuple(steps),
        )
        self.add(transition)
        return transition

    def add_step(self, step: RolloutStep, decision: PolicyDecision) -> None:
        if not isinstance(step, RolloutStep):
            raise TypeError("decision replay step must be RolloutStep")
        if not isinstance(decision, PolicyDecision):
            raise TypeError("decision replay decision must be PolicyDecision")
        if step.decision_id != decision.decision_id:
            raise ValueError("rollout step and decision ids do not match")

        pending = self._pending.get(decision.decision_id)
        if pending is None:
            pending = [decision, []]
            self._pending[decision.decision_id] = pending
        elif pending[0] is not decision:
            raise ValueError("one decision_id refers to different decisions")

        steps = pending[1]
        expected_offset = len(steps)
        if step.action_offset != expected_offset:
            raise ValueError(
                "decision replay action offsets must start at zero and be consecutive"
            )
        steps.append(step)
        if step.transition.episode_done or len(steps) == decision.chunk_size:
            self._flush_pending(decision.decision_id)

    def flush_pending(self):
        """Commit all intentionally shortened in-flight decisions."""

        flushed = tuple(
            self._flush_pending(decision_id)
            for decision_id in tuple(self._pending)
        )
        if not flushed:
            return None
        return flushed[0] if len(flushed) == 1 else flushed

    def _state_metadata(self) -> Dict[str, Any]:
        metadata = super()._state_metadata()
        metadata.update(
            {
                "pending": tuple(
                    (decision, tuple(steps))
                    for decision, steps in self._pending.values()
                ),
            }
        )
        return metadata

    def _validate_state_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        num_transitions: int,
    ) -> None:
        super()._validate_state_metadata(
            metadata,
            num_transitions=num_transitions,
        )
        pending = metadata.get("pending")
        if pending is None:
            # Backward compatibility with single-env decision replay states.
            decision = metadata.get("pending_decision")
            steps = metadata.get("pending_steps", ())
            pending = () if decision is None else ((decision, steps),)
            if decision is None and steps:
                raise ValueError("pending decision and steps must be present together")
        if not isinstance(pending, (list, tuple)):
            raise TypeError("pending decisions must be a sequence")
        decision_ids = []
        for item in pending:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError("pending entries must contain decision and steps")
            decision, steps = item
            if not isinstance(decision, PolicyDecision):
                raise TypeError("pending decision must be PolicyDecision")
            if not isinstance(steps, (list, tuple)) or not all(
                isinstance(step, RolloutStep) for step in steps
            ):
                raise TypeError("pending steps must contain RolloutStep values")
            DecisionTransition(decision=decision, steps=tuple(steps))
            decision_ids.append(decision.decision_id)
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("pending decision ids must be unique")

    def _load_state_metadata(self, metadata: Dict[str, Any]) -> None:
        super()._load_state_metadata(metadata)
        pending = metadata.get("pending")
        if pending is None:
            decision = metadata.get("pending_decision")
            steps = metadata.get("pending_steps", ())
            pending = () if decision is None else ((decision, steps),)
        self._pending = {
            decision.decision_id: [decision, list(steps)]
            for decision, steps in pending
        }
        # RLPolicyExecutor does not restore an in-flight action chunk. Commit
        # the shortened decision now so a resumed executor cannot collide with
        # or silently discard experience collected before the checkpoint.
        self.flush_pending()

    def _reset_metadata(self) -> None:
        self._pending = {}
