"""Discovery for policy-local and reusable RL policy adapters."""

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Callable, Dict, Union

from .base import BasePolicyAdapter
from .basic import BasicPolicyAdapter
from .gaussian_chunk import GaussianChunkPolicyAdapter
from .meta_policy import MetaPolicyAdapter


PolicyModule = Union[str, ModuleType]
_ADAPTER_FACTORIES: Dict[str, Callable] = {}


def register_policy_adapter(name: str, factory: Callable, *, replace: bool = False):
    if not isinstance(name, str) or not name:
        raise TypeError("adapter name must be a non-empty string")
    if not callable(factory):
        raise TypeError("policy adapter factory must be callable")
    if name in _ADAPTER_FACTORIES and not replace:
        raise ValueError(f"policy adapter {name!r} is already registered")
    _ADAPTER_FACTORIES[name] = factory
    return factory


def _build_basic(*, policy, model_components=None, **kwargs):
    del model_components
    return BasicPolicyAdapter(policy, **kwargs)


def _build_meta_policy(*, policy, model_components=None, **kwargs):
    if model_components is None or "meta_policy" not in model_components:
        raise KeyError("meta_policy adapter requires model_components meta_policy")
    meta_policy = model_components["meta_policy"]
    if getattr(meta_policy, "policy", None) is not policy:
        raise ValueError("meta_policy and model_components must share the model")
    if (
        "checkpoint_path" not in kwargs
        and model_components.get("checkpoint_path") is not None
    ):
        kwargs["checkpoint_path"] = model_components["checkpoint_path"]
    return MetaPolicyAdapter(meta_policy, **kwargs)


def _build_gaussian_chunk(*, policy, model_components=None, **kwargs):
    if model_components is None or "meta_policy" not in model_components:
        raise KeyError(
            "gaussian_chunk adapter requires model_components meta_policy"
        )
    meta_policy = model_components["meta_policy"]
    if getattr(meta_policy, "policy", None) is not policy:
        raise ValueError("meta_policy and model_components must share the model")
    if (
        "checkpoint_path" not in kwargs
        and model_components.get("checkpoint_path") is not None
    ):
        kwargs["checkpoint_path"] = model_components["checkpoint_path"]
    return GaussianChunkPolicyAdapter(meta_policy, **kwargs)


register_policy_adapter("basic", _build_basic)
register_policy_adapter("meta_policy", _build_meta_policy)
register_policy_adapter("gaussian_chunk", _build_gaussian_chunk)


def _load_policy_local_adapter(policy_module: PolicyModule):
    module = (
        importlib.import_module(policy_module)
        if isinstance(policy_module, str)
        else policy_module
    )
    if not isinstance(module, ModuleType):
        raise TypeError("policy_module must be a module or import path")
    adapter_module_name = f"{module.__name__}.rl_adapter"
    try:
        return importlib.import_module(adapter_module_name)
    except ModuleNotFoundError as exc:
        if exc.name == adapter_module_name:
            return None
        raise


def _build_local(
    adapter_module,
    model_components,
    kwargs,
    *,
    required_capabilities,
):
    local_kwargs = dict(kwargs)
    if required_capabilities:
        local_kwargs["required_capabilities"] = required_capabilities
    builder = getattr(adapter_module, "build_rl_adapter", None)
    if callable(builder):
        return builder(
            model_components=model_components,
            **local_kwargs,
        )
    adapter_class = getattr(adapter_module, "RLPolicyAdapter", None)
    if adapter_class is not None:
        return adapter_class(
            model_components=model_components,
            **local_kwargs,
        )
    raise AttributeError(
        f"{adapter_module.__name__} must define build_rl_adapter or RLPolicyAdapter"
    )


def build_policy_adapter(
    policy_module: PolicyModule,
    model_components: Mapping,
    *,
    adapter: str = "auto",
    fallback_adapter=None,
    required_capabilities=(),
    **kwargs,
) -> BasePolicyAdapter:
    """Build an adapter, preferring ``policy/<name>/rl_adapter.py`` in auto mode."""

    if not isinstance(model_components, Mapping):
        raise TypeError("model_components must be a mapping")
    if "model" not in model_components:
        raise KeyError("model_components must contain model")
    if not isinstance(adapter, str) or not adapter:
        raise TypeError("adapter must be a non-empty string")
    if fallback_adapter is not None:
        if not isinstance(fallback_adapter, str) or not fallback_adapter:
            raise TypeError("fallback_adapter must be a non-empty string or None")
        if adapter != "auto":
            raise ValueError("fallback_adapter is only valid when adapter='auto'")
        if fallback_adapter in {"auto", "policy"}:
            raise ValueError(
                "fallback_adapter must name a registered generic adapter"
            )

    result = None
    if adapter in {"auto", "policy"}:
        local_module = _load_policy_local_adapter(policy_module)
        if local_module is not None:
            result = _build_local(
                local_module,
                model_components,
                kwargs,
                required_capabilities=required_capabilities,
            )
        elif adapter == "policy":
            module_name = (
                policy_module if isinstance(policy_module, str) else policy_module.__name__
            )
            raise ModuleNotFoundError(f"{module_name}.rl_adapter was not found")

    if result is None:
        if adapter == "auto" and fallback_adapter is not None:
            generic_name = fallback_adapter
        elif adapter == "auto" and "meta_policy" in model_components:
            generic_name = "meta_policy"
        else:
            generic_name = "basic" if adapter == "auto" else adapter
        try:
            factory = _ADAPTER_FACTORIES[generic_name]
        except KeyError as exc:
            available = ", ".join(sorted(_ADAPTER_FACTORIES))
            raise ValueError(
                f"unknown policy adapter {generic_name!r}; available: {available}"
            ) from exc
        result = factory(
            policy=model_components["model"],
            model_components=model_components,
            **kwargs,
        )

    if not isinstance(result, BasePolicyAdapter):
        raise TypeError("policy adapter factory must return BasePolicyAdapter")
    result.set_checkpoint_source(model_components.get("checkpoint_path"))
    result.require_capabilities(required_capabilities)
    return result
