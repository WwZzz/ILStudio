"""Algorithm contracts separated from policy-specific optimizer mechanics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from rl.buffer import BaseBuffer, ReplayBuffer
from rl.policy_adapter import BasePolicyAdapter


@dataclass(frozen=True)
class AlgorithmOutput:
    """Loss/payload produced by RL math before optimizer stepping."""

    loss: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True)
class AlgorithmUpdateResult:
    metrics: Dict[str, Any]
    updated: bool

    def __post_init__(self):
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "updated", bool(self.updated))


class BaseRLAlgorithm(ABC):
    """Define RL loss semantics while delegating policy gradient mechanics."""

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        required_capabilities: Iterable[str] = ("action",),
        required_buffer_type: Optional[str] = None,
    ) -> None:
        capabilities = frozenset(required_capabilities)
        if not all(isinstance(item, str) and item for item in capabilities):
            raise TypeError("required capabilities must be non-empty strings")
        if required_buffer_type not in {None, "rollout", "replay"}:
            raise ValueError("required_buffer_type must be rollout, replay, or None")
        self.required_capabilities = capabilities
        self.required_buffer_type = required_buffer_type

    def validate(self, policy_adapter: BasePolicyAdapter, buffer: BaseBuffer) -> None:
        if not isinstance(policy_adapter, BasePolicyAdapter):
            raise TypeError("policy_adapter must inherit BasePolicyAdapter")
        if not isinstance(buffer, BaseBuffer):
            raise TypeError("buffer must inherit BaseBuffer")
        policy_adapter.require_capabilities(self.required_capabilities)
        if (
            self.required_buffer_type is not None
            and buffer.buffer_type != self.required_buffer_type
        ):
            raise ValueError(
                f"algorithm requires a {self.required_buffer_type} buffer, "
                f"got {buffer.buffer_type}"
            )

    @abstractmethod
    def compute_update(
        self,
        batch,
        *,
        policy_adapter: BasePolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AlgorithmOutput:
        """Compute algorithm losses and metrics for one update batch."""

    def update(
        self,
        batch,
        *,
        policy_adapter: BasePolicyAdapter,
        trainer_adapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AlgorithmUpdateResult:
        output = self.compute_update(
            batch,
            policy_adapter=policy_adapter,
            context=context,
        )
        if not isinstance(output, AlgorithmOutput):
            raise TypeError("compute_update must return AlgorithmOutput")
        trainer_result = trainer_adapter.step(
            output,
            policy_adapter=policy_adapter,
            context=context,
        )
        metrics = dict(output.metrics)
        post_step = output.payload.get("post_step")
        if trainer_result.updated and post_step is not None:
            if not isinstance(post_step, str) or not post_step:
                raise TypeError("algorithm post_step payload must be a non-empty string")
            post_context = dict(context or {})
            post_context.update(dict(output.payload.get("post_step_context", {})))
            policy_adapter.algorithm_post_step(
                post_step,
                context=post_context,
            )
        collisions = set(metrics).intersection(trainer_result.metrics)
        if collisions:
            raise KeyError(
                "algorithm and policy trainer metrics collide: "
                f"{', '.join(sorted(collisions))}"
            )
        metrics.update(trainer_result.metrics)
        return AlgorithmUpdateResult(
            metrics=metrics,
            updated=trainer_result.updated,
        )

    def iter_update_batches(
        self,
        buffer: BaseBuffer,
        *,
        batch_size: int,
        num_updates: int,
        rng: np.random.Generator,
    ):
        """Default exact-count sampler; algorithms may override epoch semantics."""

        if len(buffer) == 0:
            return
        for _ in range(num_updates):
            size = min(batch_size, len(buffer))
            if isinstance(buffer, ReplayBuffer):
                yield buffer.sample(size, replace=False)
            else:
                indices = rng.choice(len(buffer), size=size, replace=False)
                yield buffer.get_batch(indices)

    def state_dict(self) -> Dict[str, Any]:
        return {"version": self.STATE_VERSION}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("algorithm state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported algorithm state version")
