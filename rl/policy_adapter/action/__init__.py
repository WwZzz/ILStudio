"""Configurable action semantics for the default policy adapter."""

import importlib
from collections.abc import Mapping

from .base import ActionAdapter
from .gaussian_chunk import GaussianChunkActionAdapter
from .standard import (
    CategoricalActionAdapter,
    GaussianActionAdapter,
    NativeActionAdapter,
)


def build_action_adapter(spec, *, meta_policy):
    if isinstance(spec, ActionAdapter):
        if spec.meta_policy is not meta_policy:
            raise ValueError("action adapter and policy adapter must share MetaPolicy")
        return spec
    if spec is None:
        action_space = getattr(meta_policy.policy.config, "rl_action_space", None)
        adapter_type = (
            CategoricalActionAdapter
            if action_space == "discrete"
            else GaussianActionAdapter
        )
        return adapter_type(meta_policy)
    if not isinstance(spec, Mapping):
        raise TypeError("action_adapter must be an ActionAdapter or mapping")
    unknown = set(spec) - {"type", "args"}
    if unknown:
        raise ValueError("unknown action_adapter keys: " + ", ".join(sorted(unknown)))
    adapter_type = spec.get("type")
    if not isinstance(adapter_type, str) or "." not in adapter_type:
        raise TypeError("action_adapter.type must be a full import path")
    module_name, symbol_name = adapter_type.rsplit(".", 1)
    adapter_class = getattr(importlib.import_module(module_name), symbol_name)
    args = spec.get("args", {})
    if not isinstance(args, Mapping):
        raise TypeError("action_adapter.args must be a mapping")
    result = adapter_class(meta_policy, **dict(args))
    if not isinstance(result, ActionAdapter):
        raise TypeError("action_adapter.type must construct ActionAdapter")
    return result


__all__ = [
    "ActionAdapter",
    "CategoricalActionAdapter",
    "GaussianActionAdapter",
    "GaussianChunkActionAdapter",
    "NativeActionAdapter",
    "build_action_adapter",
]
