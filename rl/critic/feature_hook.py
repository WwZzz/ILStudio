"""Critic-owned feature extraction from arbitrary policy modules."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager

import torch
import torch.nn as nn

def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path, including numeric Sequential indices."""

    if not isinstance(root, nn.Module):
        raise TypeError("feature hook root must be a torch module")
    if not isinstance(path, str):
        raise TypeError("feature hook module_path must be a string")
    current = root
    if path:
        for part in path.split("."):
            if isinstance(current, (nn.Sequential, nn.ModuleList)) and part.isdigit():
                current = current[int(part)]
            elif isinstance(current, nn.ModuleDict) and part in current:
                current = current[part]
            else:
                current = getattr(current, part)
            if not isinstance(current, nn.Module):
                raise TypeError(
                    f"feature hook path {path!r} resolves through a non-module"
                )
    return current


def _select_output(value, path):
    if path in {None, ""}:
        return value
    for part in str(path).split("."):
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, (tuple, list)) and part.lstrip("-").isdigit():
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def _pool(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return value
    if mode == "cls":
        if value.ndim < 3:
            raise ValueError("cls pooling requires [batch, token, feature]")
        return value[:, 0]
    if mode == "mean_tokens":
        if value.ndim < 3:
            raise ValueError("mean_tokens pooling requires a token dimension")
        return value.mean(dim=1)
    if mode == "mean_spatial":
        if value.ndim < 4:
            raise ValueError("mean_spatial pooling requires [batch, channel, ...]")
        return value.flatten(start_dim=2).mean(dim=-1)
    if mode == "flatten":
        if value.ndim < 2:
            raise ValueError("flatten pooling requires a batch dimension")
        return value.flatten(start_dim=1)
    raise ValueError(
        "feature hook pool must be none, cls, mean_tokens, mean_spatial, or flatten"
    )


class ModuleOutputHook:
    """Capture and normalize one configured module output."""

    def __init__(
        self,
        module: nn.Module,
        *,
        output_path=None,
        pool="none",
        expected_dim=None,
        occurrence=-1,
        detach=True,
    ):
        if not isinstance(module, nn.Module):
            raise TypeError("feature hook target must be a torch module")
        if expected_dim is not None and (
            isinstance(expected_dim, bool)
            or not isinstance(expected_dim, int)
            or expected_dim <= 0
        ):
            raise ValueError("feature hook expected_dim must be a positive integer")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int):
            raise TypeError("feature hook occurrence must be an integer")
        if not isinstance(detach, bool):
            raise TypeError("feature hook detach must be bool")
        self.module = module
        self.output_path = output_path
        self.pool = pool
        self.expected_dim = expected_dim
        self.occurrence = occurrence
        self.detach = detach
        self._captured = None

    @contextmanager
    def capture(self):
        if self._captured is not None:
            raise RuntimeError("feature hook capture contexts cannot be nested")
        self._captured = []

        def record(_module, _inputs, output):
            self._captured.append(output)

        handle = self.module.register_forward_hook(record)
        try:
            yield self
        except Exception:
            self._captured = None
            raise
        finally:
            handle.remove()

    def result(self):
        captured = self._captured
        self._captured = None
        if not captured:
            raise RuntimeError("configured feature hook did not observe a forward call")
        try:
            value = captured[self.occurrence]
        except IndexError as exc:
            raise IndexError("feature hook occurrence is outside captured calls") from exc
        value = _select_output(value, self.output_path)
        if not torch.is_tensor(value):
            raise TypeError("configured feature hook output must resolve to a tensor")
        value = _pool(value, self.pool)
        if value.ndim != 2:
            raise ValueError("pooled feature hook output must have shape [batch, dim]")
        if self.expected_dim is not None and value.shape[-1] != self.expected_dim:
            raise ValueError(
                f"feature hook expected dim {self.expected_dim}, got {value.shape[-1]}"
            )
        return value.detach() if self.detach else value


class PolicyFeatureExtractor:
    """Capture critic inputs while triggering the policy adapter forward path."""

    def __init__(self, policy_adapter, hooks):
        policy = getattr(policy_adapter, "policy", None)
        if not isinstance(policy, nn.Module):
            raise TypeError("policy feature hooks require a torch policy module")
        feature_forward = getattr(policy_adapter, "feature_forward", None)
        if not callable(feature_forward):
            raise TypeError("policy adapter must provide feature_forward()")
        if not isinstance(hooks, Mapping) or not hooks:
            raise TypeError("critic feature hooks must be a non-empty mapping")
        self.feature_forward = feature_forward
        self.hooks = {}
        for name, raw_spec in hooks.items():
            if not isinstance(name, str) or not name:
                raise TypeError("critic feature hook names must be non-empty strings")
            if not isinstance(raw_spec, Mapping):
                raise TypeError(f"critic feature hook {name!r} must be a mapping")
            spec = dict(raw_spec)
            try:
                module_path = spec.pop("module_path")
            except KeyError as exc:
                raise KeyError(
                    f"critic feature hook {name!r} is missing module_path"
                ) from exc
            self.hooks[name] = ModuleOutputHook(
                resolve_module(policy, module_path),
                **spec,
            )

    def __call__(self, observations, *, context=None):
        with ExitStack() as stack:
            for hook in self.hooks.values():
                stack.enter_context(hook.capture())
            base_features = self.feature_forward(observations, context=context)
        if base_features is None:
            features = {}
        elif isinstance(base_features, Mapping):
            features = dict(base_features)
        else:
            raise TypeError("policy feature_forward must return a mapping or None")
        for name, hook in self.hooks.items():
            if name in features:
                raise ValueError(
                    f"critic feature hook {name!r} conflicts with feature_forward output"
                )
            features[name] = hook.result()
        return features


__all__ = [
    "ModuleOutputHook",
    "PolicyFeatureExtractor",
    "resolve_module",
]
