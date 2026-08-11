"""Callable algorithm adapter for lightweight and policy-local RL variants."""

import inspect
from collections.abc import Mapping
from typing import Any, Callable, Optional

from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm


def _call_compute(function, *, policy_adapter, batch, context):
    kwargs = {
        "policy_adapter": policy_adapter,
        "batch": batch,
        "context": context,
    }
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(**kwargs)


class BasicRLAlgorithm(BaseRLAlgorithm):
    """Wrap a compute callable in the stable algorithm contract."""

    def __init__(self, compute_fn: Callable, **kwargs) -> None:
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        super().__init__(**kwargs)
        self.compute_fn = compute_fn

    @staticmethod
    def _normalize_output(result) -> AlgorithmOutput:
        if isinstance(result, AlgorithmOutput):
            return result
        if not isinstance(result, Mapping):
            raise TypeError("algorithm compute_fn must return a mapping or AlgorithmOutput")
        metrics = dict(result.get("metrics", {}))
        metrics.update(
            {
                key: value
                for key, value in result.items()
                if key not in {"loss", "metrics", "payload"}
            }
        )
        return AlgorithmOutput(
            loss=result.get("loss"),
            metrics=metrics,
            payload=dict(result.get("payload", {})),
        )

    def compute_update(
        self,
        batch,
        *,
        policy_adapter: MetaPolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AlgorithmOutput:
        result = _call_compute(
            self.compute_fn,
            policy_adapter=policy_adapter,
            batch=batch,
            context=context,
        )
        return self._normalize_output(result)
