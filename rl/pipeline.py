"""Declarative component graph for assembling ILStudio RL pipelines."""

import importlib
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict


_ALLOWED_TOP_LEVEL = {
    "version",
    "name",
    "components",
    "entrypoint",
    "run",
    "metadata",
}
_ALLOWED_COMPONENT_KEYS = {"type", "args"}


def import_symbol(target: str):
    """Import one class or factory from ``module.symbol`` or ``module:symbol``."""

    if not isinstance(target, str) or not target:
        raise TypeError("component type must be a non-empty import string")
    if ":" in target:
        module_name, symbol_path = target.split(":", 1)
    else:
        try:
            module_name, symbol_path = target.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(
                f"component type must include a module and symbol: {target!r}"
            ) from exc
    try:
        value = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"failed to import component module {module_name!r} for {target!r}"
        ) from exc
    for part in symbol_path.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as exc:
            raise ImportError(f"component target {target!r} was not found") from exc
    if not callable(value):
        raise TypeError(f"component target {target!r} must be callable")
    return value


def validate_rl_config(config: Mapping[str, Any], *, import_targets: bool = False):
    """Validate graph structure without instantiating models or environments."""

    if not isinstance(config, Mapping):
        raise TypeError("RL config must be a mapping")
    unknown = set(config) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(f"unknown RL config keys: {', '.join(sorted(unknown))}")
    if config.get("version") != 1:
        raise ValueError("RL config version must be 1")

    components = config.get("components")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("RL config components must be a non-empty mapping")
    normalized_components = {}
    for name, spec in components.items():
        if not isinstance(name, str) or not name:
            raise TypeError("component names must be non-empty strings")
        if not isinstance(spec, Mapping):
            raise TypeError(f"component {name!r} must be a mapping")
        unknown_spec = set(spec) - _ALLOWED_COMPONENT_KEYS
        if unknown_spec:
            raise ValueError(
                f"unknown keys for component {name!r}: "
                f"{', '.join(sorted(unknown_spec))}"
            )
        target = spec.get("type")
        if not isinstance(target, str) or not target:
            raise TypeError(f"component {name!r} type must be an import string")
        args = spec.get("args", {})
        if not isinstance(args, Mapping):
            raise TypeError(f"component {name!r} args must be a mapping")
        if import_targets:
            import_symbol(target)
        normalized_components[name] = {
            "type": target,
            "args": deepcopy(dict(args)),
        }

    entrypoint = config.get("entrypoint")
    if not isinstance(entrypoint, str) or entrypoint not in normalized_components:
        raise ValueError("entrypoint must name one configured component")
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        raise TypeError("RL config run must be a mapping")

    normalized = deepcopy(dict(config))
    normalized["components"] = normalized_components
    normalized["entrypoint"] = entrypoint
    normalized["run"] = deepcopy(dict(run))
    return normalized


def _lookup_reference(reference: str, components: Mapping[str, Any]):
    if not isinstance(reference, str) or not reference:
        raise TypeError("$ref must be a non-empty string")
    parts = reference.split(".")
    component_name = parts.pop(0)
    if component_name not in components:
        raise ValueError(
            f"unknown or forward component reference {reference!r}; "
            "components may only reference earlier graph entries"
        )
    value = components[component_name]
    for part in parts:
        if isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"reference {reference!r} has no key {part!r}")
            value = value[part]
        else:
            try:
                value = getattr(value, part)
            except AttributeError as exc:
                raise ValueError(
                    f"reference {reference!r} has no attribute {part!r}"
                ) from exc
    return value


def _resolve_refs(value, components):
    if isinstance(value, Mapping):
        if "$ref" in value:
            if set(value) != {"$ref"}:
                raise ValueError("a $ref mapping cannot contain other keys")
            return _lookup_reference(value["$ref"], components)
        return {key: _resolve_refs(item, components) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(item, components) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_refs(item, components) for item in value)
    return value


class BuiltRLPipeline:
    """A built graph whose configured entrypoint owns the training lifecycle."""

    def __init__(self, *, config, components):
        self.config = config
        self.components = dict(components)
        self.entry = self.components[config["entrypoint"]]
        if not callable(getattr(self.entry, "run", None)):
            raise TypeError("RL pipeline entrypoint must provide run()")
        self._closed = False

    def run(self):
        if self._closed:
            raise RuntimeError("RL pipeline is closed")
        return self.entry.run(**self.config["run"])

    def close(self):
        if self._closed:
            return
        close = getattr(self.entry, "close", None)
        if callable(close):
            close()
        self._closed = True


def build_rl_pipeline(config: Mapping[str, Any]) -> BuiltRLPipeline:
    """Instantiate an ordered component graph with explicit dependency refs."""

    config = validate_rl_config(config, import_targets=True)
    components: Dict[str, Any] = {}
    try:
        for name, spec in config["components"].items():
            factory = import_symbol(spec["type"])
            args = _resolve_refs(spec["args"], components)
            components[name] = factory(**args)
        return BuiltRLPipeline(config=config, components=components)
    except Exception:
        seen = set()
        for component in reversed(tuple(components.values())):
            if id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if callable(close):
                close()
        raise
