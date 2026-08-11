"""Discovery for policy-local and reusable RL policy adapters."""

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Callable, Dict, Union

from .base import MetaPolicyAdapter
from .base import build_rl_adapter as build_base_policy_adapter


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


def _build_base(*, policy, model_components=None, **kwargs):
    if model_components is None:
        model_components = {"model": policy}
    return build_base_policy_adapter(
        model_components=model_components,
        **kwargs,
    )


register_policy_adapter("base", _build_base)


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
    raise AttributeError(
        f"{adapter_module.__name__} must define build_rl_adapter"
    )


def build_policy_adapter(
    policy_module: PolicyModule,
    model_components: Mapping,
    *,
    adapter: str = "auto",
    fallback_adapter=None,
    required_capabilities=(),
    **kwargs,
) -> MetaPolicyAdapter:
    """Build the meta contract, using ``BasePolicyAdapter`` by configuration."""

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
    if adapter in {"auto", "policy"} and not (
        adapter == "auto" and fallback_adapter is not None
    ):
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
        else:
            generic_name = "base" if adapter == "auto" else adapter
        try:
            factory = _ADAPTER_FACTORIES[generic_name]
        except KeyError as exc:
            available = ", ".join(sorted(_ADAPTER_FACTORIES))
            raise ValueError(
                f"unknown policy adapter {generic_name!r}; available: {available}"
            ) from exc
        factory_kwargs = dict(kwargs)
        if generic_name == "base":
            factory_kwargs["required_capabilities"] = required_capabilities
        result = factory(
            policy=model_components["model"],
            model_components=model_components,
            **factory_kwargs,
        )

    if not isinstance(result, MetaPolicyAdapter):
        raise TypeError("policy adapter factory must return MetaPolicyAdapter")
    result.set_checkpoint_source(model_components.get("checkpoint_path"))
    result.require_capabilities(required_capabilities)
    return result
