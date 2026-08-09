"""On-policy rollout buffer."""

from typing import Any, Dict, Optional, Sequence, Tuple

from rl.base import MetaTransition, PolicyDecision, Rollout, RolloutStep

from .base import BaseBuffer, TransitionStorage


class RolloutBuffer(BaseBuffer):
    """Finite rollout generation that never overwrites collected experience."""

    def __init__(self, capacity: Optional[int] = None) -> None:
        super().__init__(TransitionStorage(capacity=capacity, overwrite=False))
        self._sealed = False
        self._policy_version = None
        self._trajectory_ends = []
        self._rollout_steps = []
        self._decisions = {}

    @property
    def buffer_type(self) -> str:
        return "rollout"

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def policy_version(self):
        return self._policy_version

    @property
    def rollout(self):
        if not self._rollout_steps or all(step is None for step in self._rollout_steps):
            return None
        if any(step is None for step in self._rollout_steps):
            raise RuntimeError("rollout buffer contains mixed provenance")
        return Rollout(
            steps=tuple(self._rollout_steps),
            decisions=tuple(self._decisions.values()),
        )

    def add_step(self, step: RolloutStep, decision: PolicyDecision) -> None:
        if not isinstance(step, RolloutStep):
            raise TypeError("rollout buffer step must be RolloutStep")
        if not isinstance(decision, PolicyDecision):
            raise TypeError("rollout buffer decision must be PolicyDecision")
        if step.decision_id != decision.decision_id:
            raise ValueError("rollout step and decision ids do not match")
        existing = self._decisions.get(decision.decision_id)
        if existing is not None and existing is not decision:
            raise ValueError("one decision_id refers to different decisions")
        index = len(self)
        self.add(step.transition)
        self._rollout_steps[index] = step
        self._decisions[decision.decision_id] = decision

    def _validate_extend(self, transitions: Sequence[MetaTransition]) -> None:
        super()._validate_extend(transitions)
        if self._sealed:
            raise RuntimeError("rollout buffer is sealed")

        versions = {
            transition.policy_info["policy_version"]
            for transition in transitions
            if "policy_version" in transition.policy_info
        }
        if len(versions) > 1:
            raise ValueError("rollout contains more than one policy version")
        if (
            self._policy_version is not None
            and versions
            and next(iter(versions)) != self._policy_version
        ):
            raise ValueError(
                "rollout policy version does not match the current generation"
            )

    def _after_extend(self, transitions, evicted) -> None:
        if evicted:
            raise RuntimeError("rollout storage must never evict transitions")

        start_index = len(self) - len(transitions)
        for offset, transition in enumerate(transitions):
            if transition.episode_done:
                self._trajectory_ends.append(start_index + offset + 1)

        if self._policy_version is None:
            for transition in transitions:
                if "policy_version" in transition.policy_info:
                    self._policy_version = transition.policy_info["policy_version"]
                    break
        self._rollout_steps.extend([None] * len(transitions))

    def get_batch(self, indices):
        batch = super().get_batch(indices)
        selected_steps = tuple(
            self._rollout_steps[int(index)] for index in batch.indices
        )
        if not selected_steps or any(step is None for step in selected_steps):
            return batch
        selected_ids = {step.decision_id for step in selected_steps}
        selected_decisions = tuple(
            decision
            for decision_id, decision in self._decisions.items()
            if decision_id in selected_ids
        )
        rollout = Rollout(steps=selected_steps, decisions=selected_decisions)
        return type(batch)(
            transitions=batch.transitions,
            indices=batch.indices,
            rollout=rollout,
        )

    def seal(self):
        self._sealed = True
        return self

    def trajectory_ranges(
        self,
        *,
        include_incomplete: bool = False,
    ) -> Tuple[Tuple[int, int], ...]:
        ranges = []
        start = 0
        for end in self._trajectory_ends:
            ranges.append((start, end))
            start = end
        if include_incomplete and start < len(self):
            ranges.append((start, len(self)))
        return tuple(ranges)

    def _state_metadata(self) -> Dict[str, Any]:
        return {
            "sealed": self._sealed,
            "policy_version": self._policy_version,
            "trajectory_ends": list(self._trajectory_ends),
            "rollout_steps": list(self._rollout_steps),
            "decisions": list(self._decisions.values()),
        }

    def _validate_state_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        num_transitions: int,
    ) -> None:
        if not isinstance(metadata.get("sealed"), bool):
            raise TypeError("rollout sealed metadata must be a bool")
        trajectory_ends = metadata.get("trajectory_ends")
        if not isinstance(trajectory_ends, (list, tuple)):
            raise TypeError("trajectory_ends must be a list or tuple")

        previous = 0
        for end in trajectory_ends:
            if not isinstance(end, int):
                raise TypeError("trajectory end indices must be integers")
            if end <= previous or end > num_transitions:
                raise ValueError("invalid rollout trajectory end index")
            previous = end
        rollout_steps = metadata.get("rollout_steps")
        decisions = metadata.get("decisions")
        if rollout_steps is not None:
            if not isinstance(rollout_steps, (list, tuple)):
                raise TypeError("rollout_steps must be a list or tuple")
            if len(rollout_steps) != num_transitions:
                raise ValueError("rollout_steps must align with stored transitions")
            if not all(
                step is None or isinstance(step, RolloutStep)
                for step in rollout_steps
            ):
                raise TypeError("rollout_steps must contain RolloutStep or None")
        if decisions is not None and not all(
            isinstance(decision, PolicyDecision) for decision in decisions
        ):
            raise TypeError("decisions must contain PolicyDecision values")

    def _load_state_metadata(self, metadata: Dict[str, Any]) -> None:
        self._sealed = metadata["sealed"]
        self._policy_version = metadata.get("policy_version")
        self._trajectory_ends = list(metadata["trajectory_ends"])
        self._rollout_steps = list(
            metadata.get("rollout_steps", [None] * len(self))
        )
        self._decisions = {
            decision.decision_id: decision
            for decision in metadata.get("decisions", ())
        }

    def _reset_metadata(self) -> None:
        self._sealed = False
        self._policy_version = None
        self._trajectory_ends = []
        self._rollout_steps = []
        self._decisions = {}
