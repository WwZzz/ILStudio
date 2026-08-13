"""Declarative component graph for assembling ILStudio RL pipelines."""

import importlib
from collections.abc import Mapping, Sequence
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
_BASE_COMPONENT_NAMES = frozenset(
    {
        "policy_components",
        "env",
        "envs",
        "env_runner",
        "policy_adapter",
        "offline_data",
        "offline_buffer",
        "online_buffer",
        "executor",
        "buffer",
        "rewards",
        "collector",
        "critic",
        "algorithm",
        "objective_builder",
        "optimizer",
        "trainer_adapter",
        "runner_config",
        "rl_runner",
    }
)


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


def _configured_type(config):
    if isinstance(config, Mapping):
        return config.get("type")
    return getattr(config, "type", None)


def _validate_observation_contract(env_config, critic_spec):
    """Fail before construction when both components declare static fields."""

    if env_config is None or critic_spec is None:
        return
    env_type = _configured_type(env_config)
    if not isinstance(env_type, str) or not env_type:
        return
    # Legacy short names resolve through a module-level ``create_env`` factory,
    # so there is no statically declared environment class to inspect.
    if "." not in env_type:
        return
    env_class = import_symbol(env_type)
    critic_class = import_symbol(critic_spec["type"])
    available = getattr(env_class, "observation_fields", None)
    required = getattr(critic_class, "required_observation_fields", None)
    if available is None or required is None:
        return
    available = frozenset(available)
    required = frozenset(required)
    missing = required - available
    if missing:
        raise ValueError(
            f"critic {critic_spec['type']} requires observation fields "
            f"{sorted(required)}, but environment {env_type} declares "
            f"{sorted(available)}; missing {sorted(missing)}. Select a "
            "compatible critic (Gym state tasks can use -a ppo_state or "
            "--critic state_value)."
        )


def _fragment(value, *, label, allowed):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    value = deepcopy(dict(value))
    if value.get("version") != 1:
        raise ValueError(f"{label} config version must be 1")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(
            f"unknown {label} config keys: {', '.join(sorted(unknown))}"
        )
    return value


def _component_spec(value, *, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a component mapping")
    unknown = set(value) - {"type", "args"}
    if unknown:
        raise ValueError(f"unknown {label} keys: {', '.join(sorted(unknown))}")
    target = value.get("type")
    if not isinstance(target, str) or not target:
        raise TypeError(f"{label} type must be a non-empty import string")
    args = value.get("args", {})
    if not isinstance(args, Mapping):
        raise TypeError(f"{label} args must be a mapping")
    return {"type": target, "args": deepcopy(dict(args))}


def _with_dependencies(spec, dependencies):
    spec = deepcopy(spec)
    spec["args"].update(dependencies)
    return spec


def _policy_adapter_spec(algorithm):
    config = algorithm.get("policy_adapter", {})
    if not isinstance(config, Mapping):
        raise TypeError("algorithm policy_adapter must be a mapping")
    unknown = set(config) - {
        "fallback_adapter",
        "required_capabilities",
        "args",
    }
    if unknown:
        raise ValueError(
            "unknown algorithm policy_adapter keys: "
            + ", ".join(sorted(unknown))
        )
    adapter_args = config.get("args", {})
    if not isinstance(adapter_args, Mapping):
        raise TypeError("algorithm policy_adapter args must be a mapping")
    result = {
        "policy_components": {"$ref": "policy_components"},
        "adapter": "auto",
        "required_capabilities": list(config.get("required_capabilities", ())),
        "adapter_args": deepcopy(dict(adapter_args)),
    }
    fallback = config.get("fallback_adapter")
    if fallback is not None:
        result["fallback_adapter"] = fallback
    return result


def _reward_specs(reward_configs):
    reward_components = {}
    module_refs = []
    weights = {}
    default_weight = None
    for index, raw_reward in enumerate(reward_configs):
        reward = _fragment(
            raw_reward,
            label=f"reward[{index}]",
            allowed={
                "version",
                "name",
                "metadata",
                "components",
                "weights",
                "default_weight",
            },
        )
        configured_components = reward.get("components", {})
        if not isinstance(configured_components, Mapping):
            raise TypeError("reward components must be a mapping")
        for name, raw_spec in configured_components.items():
            if not isinstance(name, str) or not name:
                raise TypeError("reward component names must be non-empty strings")
            if name in _BASE_COMPONENT_NAMES or name in reward_components:
                raise ValueError(f"duplicate or reserved reward component {name!r}")
            reward_components[name] = _component_spec(
                raw_spec,
                label=f"reward component {name!r}",
            )
            module_refs.append({"$ref": name})

        configured_weights = reward.get("weights", {})
        if not isinstance(configured_weights, Mapping):
            raise TypeError("reward weights must be a mapping")
        for key, value in configured_weights.items():
            if key in weights and weights[key] != value:
                raise ValueError(f"conflicting reward weight for {key!r}")
            weights[key] = value

        configured_default = reward.get("default_weight")
        if configured_default is not None:
            if default_weight is not None and default_weight != configured_default:
                raise ValueError("reward configs have conflicting default_weight")
            default_weight = configured_default

    composer_args = {"modules": module_refs, "weights": weights}
    if default_weight is not None:
        composer_args["default_weight"] = default_weight
    return reward_components, composer_args


def compose_rl_config(
    *,
    model_name_or_path,
    env_config=None,
    training_args,
    algorithm_config,
    objective_config=None,
    critic_config=None,
    reward_configs: Sequence = (),
    env_runner_config=None,
    runner_config,
    env_index=None,
    env_indices=None,
    runtime_args=None,
    mode="online",
    offline_config=None,
):
    """Compile native config fragments into one validated runtime graph."""

    if not isinstance(model_name_or_path, str) or not model_name_or_path:
        raise TypeError("model_name_or_path must be a non-empty checkpoint path")
    if training_args is None:
        raise TypeError("training_args is required")
    if mode not in {"online", "offline", "hybrid"}:
        raise ValueError("RL mode must be online, offline, or hybrid")
    uses_online = mode in {"online", "hybrid"}
    uses_offline = mode in {"offline", "hybrid"}
    if not isinstance(reward_configs, Sequence) or isinstance(
        reward_configs, (str, bytes)
    ):
        raise TypeError("reward_configs must be a sequence of mappings")
    if uses_online and not reward_configs:
        raise ValueError("online collection requires at least one reward config")
    if uses_online and env_config is None:
        raise ValueError("online collection requires env_config")
    if uses_online and env_runner_config is None:
        raise ValueError("online collection requires env_runner_config")
    offline_config = dict(offline_config or {})
    allowed_offline = {
        "task_config",
        "default_success",
        "reward_key",
        "step_reward",
        "success_reward",
        "failure_reward",
        "offline_ratio",
        "offline_pretrain_iterations",
        "item_type",
    }
    unknown_offline = set(offline_config) - allowed_offline
    if unknown_offline:
        raise ValueError(
            "unknown offline config keys: " + ", ".join(sorted(unknown_offline))
        )
    if uses_offline and not isinstance(offline_config.get("task_config"), Mapping):
        raise TypeError("offline and hybrid modes require task_config")
    if env_index is not None and env_indices is not None:
        raise ValueError("env_index and env_indices are mutually exclusive")

    algorithm = _fragment(
        algorithm_config,
        label="algorithm",
        allowed={
            "version",
            "name",
            "metadata",
            "type",
            "args",
            "policy_adapter",
            "buffer",
            "runner",
            "objective",
            "critic",
        },
    )
    env_runner = None
    if uses_online:
        env_runner = _fragment(
            env_runner_config,
            label="env_runner",
            allowed={
                "version",
                "name",
                "metadata",
                "runner",
                "executor",
                "collector",
            },
        )
    runner = _fragment(
        runner_config,
        label="runner",
        allowed={
            "version",
            "name",
            "metadata",
            "type",
            "config_type",
            "args",
            "collector",
            "run",
        },
    )

    algorithm_spec = _component_spec(
        {"type": algorithm.get("type"), "args": algorithm.get("args", {})},
        label="algorithm",
    )
    objective_spec = None
    if objective_config is not None:
        objective = _fragment(
            objective_config,
            label="objective",
            allowed={"version", "name", "metadata", "type", "args"},
        )
        objective_spec = _component_spec(
            {"type": objective.get("type"), "args": objective.get("args", {})},
            label="objective",
        )
        algorithm_spec = _with_dependencies(
            algorithm_spec,
            {"objective_builder": {"$ref": "objective_builder"}},
        )

    critic_spec = None
    if critic_config is not None:
        critic = _fragment(
            critic_config,
            label="critic",
            allowed={"version", "name", "metadata", "type", "args"},
        )
        critic_spec = _component_spec(
            {"type": critic.get("type"), "args": critic.get("args", {})},
            label="critic",
        )
        algorithm_spec = _with_dependencies(
            algorithm_spec,
            {"critic": {"$ref": "critic"}},
        )
    if uses_online:
        _validate_observation_contract(env_config, critic_spec)

    buffer_spec = _component_spec(algorithm.get("buffer"), label="algorithm buffer")
    env_runner_spec = None
    collector_spec = None
    executor_spec = None
    if uses_online:
        env_runner_spec = _component_spec(
            env_runner.get("runner"),
            label="env_runner runner",
        )
        collector_spec = _component_spec(
            env_runner.get("collector"),
            label="env_runner collector",
        )
    collector_override = runner.get("collector", {})
    if not isinstance(collector_override, Mapping):
        raise TypeError("runner collector override must be a mapping")
    unknown_collector = set(collector_override) - {"args"}
    if unknown_collector:
        raise ValueError(
            "unknown runner collector override keys: "
            + ", ".join(sorted(unknown_collector))
        )
    collector_override_args = collector_override.get("args", {})
    if not isinstance(collector_override_args, Mapping):
        raise TypeError("runner collector override args must be a mapping")
    if uses_online:
        collector_spec["args"].update(deepcopy(dict(collector_override_args)))
        executor_spec = _component_spec(
            env_runner.get(
                "executor",
                {"type": "rl.executor.RLPolicyExecutor", "args": {}},
            ),
            label="env_runner executor",
        )
        reward_components, reward_composer_args = _reward_specs(reward_configs)
    else:
        reward_components, reward_composer_args = {}, None

    runner_args = deepcopy(dict(algorithm.get("runner", {})))
    configured_runner_args = runner.get("args", {})
    if not isinstance(configured_runner_args, Mapping):
        raise TypeError("runner args must be a mapping")
    runner_args.update(deepcopy(dict(configured_runner_args)))
    runner_args.setdefault("seed", getattr(training_args, "seed", None))
    runner_args["mode"] = mode
    runner_args["offline_pretrain_iterations"] = int(
        offline_config.get("offline_pretrain_iterations", 0)
    )
    if mode == "offline":
        runner_args.pop("collect_steps", None)
        runner_args.pop("collect_episodes", None)

    components = {
        "policy_components": {
            "type": "rl.runtime.builder.build_policy_components_from_checkpoint",
            "args": {
                "checkpoint": model_name_or_path,
                "runtime_args": deepcopy(dict(runtime_args or {})),
            },
        },
        "policy_adapter": {
            "type": "rl.runtime.builder.build_policy_adapter_from_components",
            "args": _policy_adapter_spec(algorithm),
        },
    }
    if uses_offline:
        item_type = offline_config.get("item_type", "auto")
        if item_type not in {"auto", "transition", "decision"}:
            raise ValueError("offline item_type must be auto, transition, or decision")
        if item_type == "auto":
            item_type = (
                "decision"
                if buffer_spec["type"].rsplit(".", 1)[-1]
                == "DecisionReplayBuffer"
                else "transition"
            )
        reward_key = offline_config.get(
            "reward_key", algorithm_spec["args"].get("reward_key", "train/total")
        )
        components["offline_data"] = {
            "type": "rl.runtime.builder.build_offline_replay_dataset",
            "args": {
                "task_config": deepcopy(dict(offline_config["task_config"])),
                "training_args": training_args,
                "item_type": item_type,
                "default_success": offline_config.get("default_success", True),
                "reward_key": reward_key,
                "step_reward": offline_config.get("step_reward", 0.0),
                "success_reward": offline_config.get("success_reward", 1.0),
                "failure_reward": offline_config.get("failure_reward", 0.0),
            },
        }
        offline_buffer_spec = {
            "type": "rl.buffer.OfflineReplayBuffer",
            "args": {
                "dataset": {"$ref": "offline_data"},
                "seed": runner_args.get("seed"),
            },
        }
        if mode == "offline":
            components["buffer"] = offline_buffer_spec
        else:
            components["offline_buffer"] = offline_buffer_spec
            components["online_buffer"] = buffer_spec
            components["buffer"] = {
                "type": "rl.buffer.HybridReplayBuffer",
                "args": {
                    "offline_buffer": {"$ref": "offline_buffer"},
                    "online_buffer": {"$ref": "online_buffer"},
                    "offline_ratio": offline_config.get("offline_ratio", 0.5),
                    "seed": runner_args.get("seed"),
                },
            }
    else:
        components["buffer"] = buffer_spec

    if uses_online:
        components["executor"] = _with_dependencies(
            executor_spec,
            {"adapter": {"$ref": "policy_adapter"}},
        )
        if env_indices is None:
            components["env"] = {
                "type": "rl.runtime.builder.build_env_from_loaded_config",
                "args": {"env_config": env_config, "env_index": env_index},
            }
            components["env_runner"] = _with_dependencies(
                env_runner_spec,
                {"env": {"$ref": "env"}},
            )
        else:
            components["envs"] = {
                "type": "rl.runtime.builder.build_env_specs_from_loaded_config",
                "args": {
                    "env_config": env_config,
                    "env_indices": list(env_indices),
                },
            }
            components["env_runner"] = _with_dependencies(
                env_runner_spec,
                {"envs": {"$ref": "envs"}},
            )
    if objective_spec is not None:
        components["objective_builder"] = objective_spec
    if critic_spec is not None:
        components["critic"] = {
            "type": "rl.runtime.builder.build_torch_module",
            "args": {
                "module_type": critic_spec["type"],
                "args": critic_spec["args"],
                "device": dict(runtime_args or {}).get("device"),
                "checkpoint": model_name_or_path,
            },
        }
    components.update(reward_components)
    if uses_online:
        components.update(
            {
                "rewards": {
                    "type": "rl.reward.RewardComposer",
                    "args": reward_composer_args,
                },
                "collector": _with_dependencies(
                    collector_spec,
                    {
                        "runner": {"$ref": "env_runner"},
                        "executor": {"$ref": "executor"},
                        "buffer": {"$ref": "buffer"},
                        "reward_composer": {"$ref": "rewards"},
                    },
                ),
            }
        )
    components.update(
        {
            "algorithm": algorithm_spec,
            "optimizer": {
                "type": "rl.runtime.builder.build_optimizer_from_training",
                "args": {
                    "model": {"$ref": "policy_components.model"},
                    "training_args": training_args,
                    "extra_models": (
                        [{"$ref": "policy_adapter"}]
                        + ([{"$ref": "critic"}] if critic_spec is not None else [])
                        + [{"$ref": "algorithm"}]
                    ),
                },
            },
            "trainer_adapter": {
                "type": "rl.runtime.builder.build_trainer_adapter_from_components",
                "args": {
                    "policy_components": {"$ref": "policy_components"},
                    "policy_adapter": {"$ref": "policy_adapter"},
                    "algorithm": {"$ref": "algorithm"},
                    "adapter": "auto",
                    "optimizer": {"$ref": "optimizer"},
                },
            },
            "runner_config": {
                "type": runner.get("config_type"),
                "args": runner_args,
            },
            "rl_runner": {
                "type": runner.get("type"),
                "args": {
                    "buffer": {"$ref": "buffer"},
                    "policy_adapter": {"$ref": "policy_adapter"},
                    "algorithm": {"$ref": "algorithm"},
                    "trainer_adapter": {"$ref": "trainer_adapter"},
                    "config": {"$ref": "runner_config"},
                },
            },
        }
    )
    if uses_online:
        components["rl_runner"]["args"]["collector"] = {"$ref": "collector"}
    graph = {
        "version": 1,
        "components": components,
        "entrypoint": "rl_runner",
        "run": deepcopy(dict(runner.get("run", {}))),
        "metadata": {"seed": runner_args.get("seed"), "mode": mode},
    }
    return validate_rl_config(graph)


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
