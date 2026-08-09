"""Algorithm contracts separated from policy-specific optimizer mechanics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Real
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
        metric_totals = {}
        metric_weights = {}
        post_steps = []

        def scalar_metric(value):
            if isinstance(value, Real) and not isinstance(value, bool):
                return float(value)
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                return None
            if array.size != 1 or not np.issubdtype(array.dtype, np.number):
                return None
            return float(array.reshape(-1)[0])

        def observed_outputs():
            for output in self.iter_compute_updates(
                batch,
                policy_adapter=policy_adapter,
                context=context,
            ):
                if not isinstance(output, AlgorithmOutput):
                    raise TypeError("compute updates must yield AlgorithmOutput")
                weight = output.payload.get("metric_weight", 1.0)
                if not isinstance(weight, Real) or float(weight) <= 0:
                    raise ValueError("metric_weight must be positive")
                weight = float(weight)
                for key, value in output.metrics.items():
                    scalar = scalar_metric(value)
                    if scalar is not None:
                        metric_totals[key] = (
                            metric_totals.get(key, 0.0) + scalar * weight
                        )
                        metric_weights[key] = metric_weights.get(key, 0.0) + weight
                    elif key in metric_totals and metric_totals[key] != value:
                        raise ValueError(
                            f"non-numeric metric {key!r} changed across micro-batches"
                        )
                    else:
                        metric_totals[key] = value
                post_step = output.payload.get("post_step")
                if post_step is not None:
                    post_steps.append(
                        (post_step, dict(output.payload.get("post_step_context", {})))
                    )
                yield output

        trainer_result = trainer_adapter.step_many(
            observed_outputs(),
            policy_adapter=policy_adapter,
            context=context,
        )
        metrics = {
            key: (
                value / metric_weights[key]
                if key in metric_weights
                else value
            )
            for key, value in metric_totals.items()
        }
        if trainer_result.updated and post_steps:
            if any(item != post_steps[0] for item in post_steps[1:]):
                raise ValueError("micro-batches must request the same post_step")
            post_step, post_step_context = post_steps[0]
            if not isinstance(post_step, str) or not post_step:
                raise TypeError("algorithm post_step payload must be a non-empty string")
            post_context = dict(context or {})
            post_context.update(post_step_context)
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

    def iter_compute_updates(
        self,
        batch,
        *,
        policy_adapter: BasePolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ):
        """Yield loss graphs consumed and released by the TrainerAdapter."""

        yield self.compute_update(
            batch,
            policy_adapter=policy_adapter,
            context=context,
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
