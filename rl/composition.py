"""Compose native ILStudio RL config fragments into one runtime graph."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

from .pipeline import validate_rl_config


_BASE_COMPONENT_NAMES = frozenset(
    {
        "policy_components",
        "env",
        "env_runner",
        "policy_adapter",
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
        "required_capabilities": list(
            config.get("required_capabilities", ())
        ),
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
    env_config,
    training_args,
    algorithm_config,
    objective_config=None,
    critic_config=None,
    reward_configs: Sequence,
    env_runner_config,
    runner_config,
    env_index=None,
    runtime_args=None,
):
    """Build the declarative graph while keeping config choices independent."""

    if not isinstance(model_name_or_path, str) or not model_name_or_path:
        raise TypeError("model_name_or_path must be a non-empty checkpoint path")
    if training_args is None:
        raise TypeError("training_args is required")
    if not isinstance(reward_configs, Sequence) or isinstance(
        reward_configs, (str, bytes)
    ):
        raise TypeError("reward_configs must be a sequence of mappings")
    if not reward_configs:
        raise ValueError("at least one reward config is required")

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
    buffer_spec = _component_spec(algorithm.get("buffer"), label="algorithm buffer")
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
    collector_spec["args"].update(deepcopy(dict(collector_override_args)))
    executor_spec = _component_spec(
        env_runner.get(
            "executor",
            {"type": "rl.executor.RLPolicyExecutor", "args": {}},
        ),
        label="env_runner executor",
    )
    reward_components, reward_composer_args = _reward_specs(reward_configs)

    runner_args = deepcopy(dict(algorithm.get("runner", {})))
    configured_runner_args = runner.get("args", {})
    if not isinstance(configured_runner_args, Mapping):
        raise TypeError("runner args must be a mapping")
    runner_args.update(deepcopy(dict(configured_runner_args)))
    runner_args.setdefault("seed", getattr(training_args, "seed", None))

    components = {
        "policy_components": {
            "type": "rl.builders.build_policy_components_from_checkpoint",
            "args": {
                "checkpoint": model_name_or_path,
                "runtime_args": deepcopy(dict(runtime_args or {})),
            },
        },
        "env": {
            "type": "rl.builders.build_env_from_loaded_config",
            "args": {"env_config": env_config, "env_index": env_index},
        },
        "env_runner": _with_dependencies(
            env_runner_spec,
            {"env": {"$ref": "env"}},
        ),
        "policy_adapter": {
            "type": "rl.builders.build_policy_adapter_from_components",
            "args": _policy_adapter_spec(algorithm),
        },
        "executor": _with_dependencies(
            executor_spec,
            {"adapter": {"$ref": "policy_adapter"}},
        ),
        "buffer": buffer_spec,
    }
    if objective_spec is not None:
        components["objective_builder"] = objective_spec
    if critic_spec is not None:
        components["critic"] = {
            "type": "rl.builders.build_torch_module",
            "args": {
                "module_type": critic_spec["type"],
                "args": critic_spec["args"],
                "device": dict(runtime_args or {}).get("device"),
                "checkpoint": model_name_or_path,
            },
        }
    components.update(reward_components)
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
            "algorithm": algorithm_spec,
            "optimizer": {
                "type": "rl.builders.build_optimizer_from_training",
                "args": {
                    "model": {"$ref": "policy_components.model"},
                    "training_args": training_args,
                    "extra_models": (
                        [{"$ref": "policy_adapter"}]
                        + ([{"$ref": "critic"}] if critic_spec is not None else [])
                    ),
                },
            },
            "trainer_adapter": {
                "type": "rl.builders.build_trainer_adapter_from_components",
                "args": {
                    "policy_components": {"$ref": "policy_components"},
                    "policy_adapter": {"$ref": "policy_adapter"},
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
                    "collector": {"$ref": "collector"},
                    "buffer": {"$ref": "buffer"},
                    "policy_adapter": {"$ref": "policy_adapter"},
                    "algorithm": {"$ref": "algorithm"},
                    "trainer_adapter": {"$ref": "trainer_adapter"},
                    "config": {"$ref": "runner_config"},
                },
            },
        }
    )
    graph = {
        "version": 1,
        "components": components,
        "entrypoint": "rl_runner",
        "run": deepcopy(dict(runner.get("run", {}))),
        "metadata": {"seed": runner_args.get("seed")},
    }
    return validate_rl_config(graph)
