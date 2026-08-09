"""Generic policy adapter composable with existing ILStudio policies."""

import inspect
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Optional

import numpy as np

from benchmark.base import MetaAction, MetaObs

from .base import BasePolicyAdapter


def _call_with_supported_kwargs(function, *args, **kwargs):
    """Pass optional context flags only when a policy declares them."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **kwargs)


class BasicPolicyAdapter(BasePolicyAdapter):
    """Adapt a callable or ``select_action`` policy to RL contracts.

    ``obs_adapter`` and ``action_adapter`` are intentionally injectable.  A
    policy that needs substantial tokenization, recurrent state, value heads or
    log-probability reconstruction should instead provide
    ``policy/<name>/rl_adapter.py``.
    """

    def __init__(
        self,
        policy: Any,
        *,
        action_fn: Optional[Callable] = None,
        train_forward_fn: Optional[Callable] = None,
        algorithm_forward_fn: Optional[Callable] = None,
        algorithm_post_step_fn: Optional[Callable] = None,
        obs_adapter: Optional[Callable[[MetaObs], Any]] = None,
        action_adapter: Optional[Callable[[Any], MetaAction]] = None,
        action_kwargs: Optional[Mapping[str, Any]] = None,
        capabilities: Iterable[str] = ("action",),
    ) -> None:
        super().__init__(policy, capabilities=capabilities)
        if action_fn is not None and not callable(action_fn):
            raise TypeError("action_fn must be callable")
        if train_forward_fn is not None and not callable(train_forward_fn):
            raise TypeError("train_forward_fn must be callable")
        if algorithm_forward_fn is not None and not callable(algorithm_forward_fn):
            raise TypeError("algorithm_forward_fn must be callable")
        if algorithm_post_step_fn is not None and not callable(algorithm_post_step_fn):
            raise TypeError("algorithm_post_step_fn must be callable")
        if obs_adapter is not None and not callable(obs_adapter):
            raise TypeError("obs_adapter must be callable")
        if action_adapter is not None and not callable(action_adapter):
            raise TypeError("action_adapter must be callable")
        self.action_fn = action_fn
        self.train_forward_fn = train_forward_fn
        self.algorithm_forward_fn = algorithm_forward_fn
        self.algorithm_post_step_fn = algorithm_post_step_fn
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        self.action_kwargs = dict(action_kwargs or {})

    def _resolve_action_fn(self):
        if self.action_fn is not None:
            return self.action_fn, True
        for name in ("select_action", "act"):
            method = getattr(self.policy, name, None)
            if callable(method):
                return method, False
        if callable(self.policy):
            return self.policy, False
        raise TypeError("policy must be callable or provide select_action/act")

    def _normalize_output(self, output):
        if isinstance(output, (dict, MetaAction)):
            return output
        from rl.base import PolicyOutput

        if isinstance(output, PolicyOutput):
            return output
        if self.action_adapter is not None:
            action = self.action_adapter(output)
            if not isinstance(action, MetaAction):
                raise TypeError("action_adapter must return MetaAction")
            return action
        if isinstance(output, (np.ndarray, list, tuple)) or hasattr(output, "shape"):
            return MetaAction(action=output, **self.action_kwargs)
        raise TypeError(
            "policy output cannot be converted to MetaAction; provide action_adapter"
        )

    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
        self._validate_obs(obs)
        policy_obs = self.obs_adapter(obs) if self.obs_adapter is not None else obs
        action_fn, inject_policy = self._resolve_action_fn()
        args = (self.policy, policy_obs) if inject_policy else (policy_obs,)
        output = _call_with_supported_kwargs(
            action_fn,
            *args,
            deterministic=deterministic,
            context=context,
        )
        return self._finalize_output(self._normalize_output(output))

    def training_forward(self, batch, *, context=None):
        if self.train_forward_fn is None:
            return super().training_forward(batch, context=context)
        result = _call_with_supported_kwargs(
            self.train_forward_fn,
            self.policy,
            batch,
            context=context,
        )
        if not isinstance(result, Mapping):
            raise TypeError("train_forward_fn must return a mapping")
        return result

    def algorithm_forward(self, operation, batch, *, context=None):
        if not isinstance(operation, str) or not operation:
            raise TypeError("algorithm operation must be a non-empty string")
        if self.algorithm_forward_fn is None:
            return super().algorithm_forward(operation, batch, context=context)
        result = _call_with_supported_kwargs(
            self.algorithm_forward_fn,
            self.policy,
            operation,
            batch,
            context=context,
        )
        if not isinstance(result, Mapping):
            raise TypeError("algorithm_forward_fn must return a mapping")
        return result

    def algorithm_post_step(self, operation, *, context=None):
        if not isinstance(operation, str) or not operation:
            raise TypeError("algorithm operation must be a non-empty string")
        if self.algorithm_post_step_fn is None:
            return super().algorithm_post_step(operation, context=context)
        return _call_with_supported_kwargs(
            self.algorithm_post_step_fn,
            self.policy,
            operation,
            context=context,
        )
