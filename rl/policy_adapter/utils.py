"""Shared MetaPolicy mechanics and construction helpers for RL adapters."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Dict, Union

import numpy as np
import torch

from benchmark.base import MetaAction, MetaObs, MetaPolicy, dict2meta
from rl.base import MetaTransition

if TYPE_CHECKING:
    from .base import MetaPolicyAdapter


PolicyModule = Union[str, ModuleType]
_ADAPTER_FACTORIES: Dict[str, Callable] = {}


def model_device(model):
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return next(parameters()).device
    except StopIteration:
        return None


def move_to_device(value, device):
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def resolve_chunk_size(policy, chunk_size=None):
    if chunk_size is not None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return chunk_size
    config = getattr(policy, "config", SimpleNamespace())
    for name in ("chunk_size", "num_queries", "prediction_horizon"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError("chunk_size is absent from policy configuration")


def policy_batch(meta_policy: MetaPolicy, observations):
    samples = []
    for obs in tuple(observations):
        if not isinstance(obs, MetaObs):
            raise TypeError("policy observation must be MetaObs")
        normalized = meta_policy.state_normalizer.normalize_metaobs(
            copy.deepcopy(obs), meta_policy.ctrl_space
        )
        items = meta_policy.normed_mobs_to_samples(normalized)
        if len(items) != 1:
            raise ValueError("each MetaObs must produce exactly one policy sample")
        samples.append(items[0])
    if not samples:
        raise ValueError("observation batch cannot be empty")
    return move_to_device(
        meta_policy.meta2obs(samples),
        model_device(meta_policy.policy),
    )


def _single_step_meta_action(entry):
    if isinstance(entry, np.ndarray) and entry.dtype == object:
        values = entry.reshape(-1).tolist()
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, Mapping)):
        values = list(entry)
    else:
        values = [entry]
    if len(values) != 1:
        raise ValueError("MetaPolicy inference supports one synchronous environment")
    value = values[0]
    if isinstance(value, MetaAction):
        return value
    if isinstance(value, Mapping):
        return dict2meta(dict(value), mtype="action")
    raise TypeError("MetaPolicy inference entries must contain MetaAction mappings")


def infer_action(meta_policy: MetaPolicy, obs: MetaObs):
    with torch.inference_mode():
        entries = meta_policy.inference(copy.deepcopy(obs))
    if not isinstance(entries, (list, tuple)) or not entries:
        raise TypeError("MetaPolicy.inference must return a non-empty action list")
    steps = tuple(_single_step_meta_action(entry) for entry in entries)
    first = steps[0]
    actions = []
    for step in steps:
        if step.ctrl_space != first.ctrl_space or step.ctrl_type != first.ctrl_type:
            raise ValueError("action chunk contains inconsistent control metadata")
        value = np.asarray(step.action)
        if value.ndim != 1:
            raise ValueError("each action-list entry must contain one action vector")
        actions.append(value)
    return MetaAction(
        action=np.stack(actions).astype(np.float32, copy=False),
        ctrl_space=first.ctrl_space,
        ctrl_type=first.ctrl_type,
        gripper_continuous=first.gripper_continuous,
    )


def explore_action(meta_policy, action, *, rng, std, clip=None):
    if std == 0.0:
        return action
    normalized = meta_policy.action_normalizer.normalize(
        np.asarray(action.action).copy(), datatype="action"
    )
    normalized += rng.normal(0.0, std, size=normalized.shape).astype(normalized.dtype)
    if clip is not None:
        normalized = np.clip(normalized, *clip)
    explored = meta_policy.action_normalizer.denormalize(
        normalized, datatype="action"
    )
    return MetaAction(
        action=np.asarray(explored, dtype=np.float32),
        ctrl_space=action.ctrl_space,
        ctrl_type=action.ctrl_type,
        gripper_continuous=action.gripper_continuous,
    )


def _chunk_sample(meta_policy, chunk, *, chunk_size):
    chunk = tuple(chunk)
    if not chunk or not all(isinstance(item, MetaTransition) for item in chunk):
        raise TypeError("training chunks must contain MetaTransition values")
    actions = [np.asarray(item.action.action) for item in chunk[:chunk_size]]
    if any(action.ndim != 1 for action in actions):
        raise ValueError("training transitions must contain single-step actions")
    action_dim = actions[0].shape[0]
    if any(action.shape != (action_dim,) for action in actions):
        raise ValueError("training chunk action dimensions must agree")
    valid = len(actions)
    padded = np.zeros((chunk_size, action_dim), dtype=np.float32)
    padded[:valid] = np.asarray(actions, dtype=np.float32)
    padded = meta_policy.action_normalizer.normalize(padded, datatype="action")
    is_pad = np.ones(chunk_size, dtype=bool)
    is_pad[:valid] = False
    normalized = meta_policy.state_normalizer.normalize_metaobs(
        copy.deepcopy(chunk[0].obs), meta_policy.ctrl_space
    )
    samples = meta_policy.normed_mobs_to_samples(normalized)
    if len(samples) != 1:
        raise ValueError("each action chunk must begin with one unbatched MetaObs")
    sample = samples[0]
    sample["action"] = torch.from_numpy(np.asarray(padded)).float()
    sample["is_pad"] = torch.from_numpy(is_pad)
    return sample, valid


def native_training_forward(meta_policy, batch, *, chunk_size):
    chunks = getattr(batch, "chunks", batch)
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise TypeError("chunk training batch must provide a non-empty chunks sequence")
    prepared = [_chunk_sample(meta_policy, chunk, chunk_size=chunk_size) for chunk in chunks]
    samples = [sample for sample, _ in prepared]
    policy_batch_value = move_to_device(
        meta_policy.meta2obs(samples), model_device(meta_policy.policy)
    )
    result = meta_policy.policy(**policy_batch_value)
    if not isinstance(result, Mapping) or "loss" not in result:
        raise TypeError("native policy forward must return a mapping containing loss")
    output = dict(result)
    output["num_chunks"] = len(samples)
    output["num_actions"] = sum(valid for _, valid in prepared)
    return output


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
    from .base import build_rl_adapter as build_base_policy_adapter

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

    from .base import MetaPolicyAdapter

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
