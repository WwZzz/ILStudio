"""Bridges from declarative RL graphs to existing ILStudio config loaders."""

import importlib
from collections.abc import Mapping
from types import SimpleNamespace

from benchmark.base import MetaEnv
from benchmark.env_runner import MetaEnvSpec
from configs.loader import ConfigLoader
from policy.direct_loader import direct_policy_loader
from policy.policy_loader import (
    get_policy_trainer_class,
    load_policy_model_for_training,
)
from policy.utils import load_policy
from .pipeline import import_symbol
from data_utils.offline_rl import OfflineReplayDataset
from rl.policy_adapter import build_policy_adapter
from rl.policy_adapter.training import (
    build_grouped_trainer_adapter,
    build_trainer_adapter,
)


def _validate_override_groups(overrides):
    overrides = dict(overrides or {})
    for category, values in overrides.items():
        if not isinstance(category, str) or not isinstance(values, Mapping):
            raise TypeError("config overrides must map categories to flat mappings")
    return {category: dict(values) for category, values in overrides.items()}


def build_env_from_loaded_config(env_config, *, env_index=None) -> MetaEnv:
    """Build a benchmark env from the result of ``ConfigLoader.load_env``."""

    if isinstance(env_config, list):
        if env_index is None:
            raise ValueError(
                "multi-environment config requires an explicit env_index; "
                "parallel execution must be selected deliberately"
            )
        if not isinstance(env_index, int) or not 0 <= env_index < len(env_config):
            raise IndexError("env_index is outside the configured environment list")
        env_config = env_config[env_index]
    elif env_index not in {None, 0}:
        raise IndexError("single-environment config only supports env_index 0")

    env_type = getattr(env_config, "type", None)
    if not isinstance(env_type, str) or not env_type:
        raise TypeError("environment config type must be a non-empty string")
    module_name = (
        env_type.rsplit(".", 1)[0]
        if "." in env_type
        else f"benchmark.{env_type}"
    )
    module = importlib.import_module(module_name)
    create_env = getattr(module, "create_env", None)
    if not callable(create_env):
        raise AttributeError(f"environment module {module_name!r} has no create_env")
    result = create_env(env_config)
    if not isinstance(result, MetaEnv):
        raise TypeError("benchmark create_env must return an ILStudio MetaEnv")
    return result


def build_offline_replay_dataset(
    task_config,
    training_args,
    *,
    item_type="transition",
    default_success=True,
    reward_key="train/total",
    step_reward=0.0,
    success_reward=1.0,
    failure_reward=0.0,
):
    """Load raw task samples and attach RL-only episode semantics.

    ``normalize=False`` is deliberate: replay stores environment-space state
    and action values, and the checkpoint's MetaPolicy applies its own saved
    normalizers during policy/algorithm forwards.
    """

    from data_utils.utils import load_datasets

    datasets = load_datasets(
        training_args,
        task_config,
        save_norm=False,
        normalize=False,
    )
    return OfflineReplayDataset(
        datasets,
        item_type=item_type,
        default_success=default_success,
        reward_key=reward_key,
        step_reward=step_reward,
        success_reward=success_reward,
        failure_reward=failure_reward,
    )


def build_env_specs_from_loaded_config(env_config, *, env_indices):
    """Build lightweight specs for an explicit subset of a multi-env config."""

    configs = tuple(env_config) if isinstance(env_config, list) else (env_config,)
    if not isinstance(env_indices, (list, tuple)) or not env_indices:
        raise TypeError("env_indices must be a non-empty list or tuple")
    selected = []
    seen = set()
    for index in env_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("environment indices must be integers")
        if not 0 <= index < len(configs):
            raise IndexError("env_index is outside the configured environment list")
        if index in seen:
            raise ValueError("environment indices must be unique")
        seen.add(index)
        selected.append(configs[index])

    specs = []
    for config in selected:
        env_type = getattr(config, "type", None)
        if not isinstance(env_type, str) or not env_type:
            raise TypeError("environment config type must be a non-empty string")
        module_name = (
            env_type.rsplit(".", 1)[0]
            if "." in env_type
            else f"benchmark.{env_type}"
        )
        module = importlib.import_module(module_name)
        create_env = getattr(module, "create_env", None)
        if not callable(create_env):
            raise AttributeError(
                f"environment module {module_name!r} has no create_env"
            )
        specs.append(MetaEnvSpec(create_env, config))
    return tuple(specs)


def build_env_from_config(
    env: str,
    *,
    overrides=None,
    env_index=None,
) -> MetaEnv:
    """Build one benchmark ``MetaEnv`` through ``ConfigLoader.load_env``."""

    env_overrides = {}
    for key, value in dict(overrides or {}).items():
        normalized_key = (
            key
            if "." in key or key in {"type", "name", "envs"}
            else f"args.{key}"
        )
        env_overrides[normalized_key] = value
    loader = ConfigLoader(unknown_args={"env": env_overrides})
    env_config, _ = loader.load_env(env)
    if isinstance(env_config, list):
        if env_index is None:
            raise ValueError(
                "multi-environment config requires an explicit env_index; "
                "parallel execution must be selected deliberately"
            )
        if not isinstance(env_index, int) or not 0 <= env_index < len(env_config):
            raise IndexError("env_index is outside the configured environment list")
        env_config = env_config[env_index]
    elif env_index not in {None, 0}:
        raise IndexError("single-environment config only supports env_index 0")

    env_type = env_config.type
    if not isinstance(env_type, str) or not env_type:
        raise TypeError("environment config type must be a non-empty string")
    module_name = env_type.rsplit(".", 1)[0] if "." in env_type else f"benchmark.{env_type}"
    module = importlib.import_module(module_name)
    create_env = getattr(module, "create_env", None)
    if not callable(create_env):
        raise AttributeError(f"environment module {module_name!r} has no create_env")
    result = create_env(env_config)
    if not isinstance(result, MetaEnv):
        raise TypeError("benchmark create_env must return an ILStudio MetaEnv")
    return result


def build_policy_components_from_config(
    *,
    policy: str,
    task: str,
    training: str = "default",
    runtime_args=None,
    overrides=None,
):
    """Load a trainable policy with the same merge rules as ``train.py``."""

    runtime = {
        "is_training": True,
        "output_dir": "ckpt/rl",
        "eval_ratio": 0.0,
        "resume_from_checkpoint": None,
    }
    runtime.update(dict(runtime_args or {}))
    runtime["is_training"] = True
    args = SimpleNamespace(**runtime)
    loader = ConfigLoader(
        args=args,
        unknown_args=_validate_override_groups(overrides),
    )
    task_config, task_path = loader.load_task(task)
    policy_config, policy_path = loader.load_policy(policy)
    training_config, training_args, training_path = loader.load_training(
        training,
        hyper_args=args,
    )
    ConfigLoader.merge_all_parameters(
        task_config,
        policy_config,
        training_config,
        args,
    )
    model_components = load_policy_model_for_training(
        policy_path,
        args,
        task_config,
    )
    if not isinstance(model_components, Mapping) or "model" not in model_components:
        raise TypeError("policy loader must return a mapping containing model")

    result = dict(model_components)
    result.update(
        {
            "args": args,
            "task_config": task_config,
            "policy_config": policy_config,
            "training_config": training_config,
            "training_args": training_args,
            "policy_module": policy_config["module_path"],
            "paths": {
                "task": task_path,
                "policy": policy_path,
                "training": training_path,
            },
        }
    )
    return result


def build_policy_components_from_checkpoint(
    checkpoint: str,
    *,
    runtime_args=None,
):
    """Load one trainable local checkpoint through ILStudio's eval loader.

    Checkpoint metadata, normalizers, model configuration and device placement
    therefore remain identical to ``eval_sim.py`` and ``eval_real.py``.
    """

    if not isinstance(checkpoint, str) or not checkpoint:
        raise TypeError("checkpoint must be a non-empty path string")
    runtime = dict(runtime_args or {})
    runtime.update(
        {
            "model_name_or_path": checkpoint,
            "is_training": False,
            # Evaluation construction is intentionally reused so checkpoint
            # metadata and normalizers stay identical.  Policy-local loaders
            # may use this explicit flag to keep parameter-efficient adapters
            # trainable without changing eval_sim/eval_real behavior.
            "rl_training": True,
        }
    )
    args = SimpleNamespace(**runtime)
    meta_policy = load_policy(args)
    model = getattr(meta_policy, "policy", None)
    if model is None:
        raise TypeError("checkpoint loader must return a local MetaPolicy")

    policy_type = direct_policy_loader.detect_policy_type(checkpoint)
    return {
        "model": model,
        "meta_policy": meta_policy,
        "args": args,
        "policy_module": f"policy.{policy_type}",
        "checkpoint_path": checkpoint,
        "paths": {"checkpoint": checkpoint},
    }


def build_policy_adapter_from_components(
    policy_components: Mapping,
    *,
    adapter: str = "auto",
    fallback_adapter=None,
    required_capabilities=(),
    adapter_args=None,
):
    if not isinstance(policy_components, Mapping):
        raise TypeError("policy_components must be a mapping")
    try:
        policy_module = policy_components["policy_module"]
    except KeyError as exc:
        raise KeyError("policy_components must contain policy_module") from exc
    kwargs = dict(adapter_args or {})
    if fallback_adapter is not None:
        kwargs["fallback_adapter"] = fallback_adapter
    return build_policy_adapter(
        policy_module,
        policy_components,
        adapter=adapter,
        required_capabilities=required_capabilities,
        **kwargs,
    )


def build_optimizer(
    model,
    *,
    extra_models=(),
    optimizer_type: str = "torch.optim.AdamW",
    args=None,
    trainable_only: bool = True,
):
    """Build a generic optimizer; policy-specific parameter groups stay pluggable."""

    models = (model, *tuple(extra_models))
    params = []
    seen = set()
    for item in models:
        parameters = getattr(item, "parameters", None)
        if not callable(parameters):
            raise TypeError("optimizer models must provide parameters()")
        for parameter in parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                params.append(parameter)
    params = tuple(params)
    if trainable_only:
        params = tuple(
            parameter
            for parameter in params
            if bool(getattr(parameter, "requires_grad", True))
        )
    if not params:
        raise ValueError("optimizer received no trainable parameters")
    optimizer_class = import_symbol(optimizer_type)
    return optimizer_class(params, **dict(args or {}))


_TRAINING_OPTIMIZERS = {
    "adamw_torch": "torch.optim.AdamW",
}


def build_optimizer_from_training(
    model,
    training_args,
    *,
    extra_models=(),
    optimizer_type=None,
    trainable_only: bool = True,
):
    """Map the optimizer subset of ILStudio training config into torch.

    The full Hugging Face Trainer lifecycle is deliberately not reused by RL.
    Rollout batching, update counts, schedulers, logging and checkpoint state
    remain owned by RLRunner and its composed components.
    """

    if training_args is None:
        raise TypeError("training_args is required")
    if optimizer_type is None:
        optimizer_name = getattr(training_args, "optim", "adamw_torch")
        optimizer_name = getattr(optimizer_name, "value", optimizer_name)
        try:
            optimizer_type = _TRAINING_OPTIMIZERS[str(optimizer_name)]
        except KeyError as exc:
            supported = ", ".join(sorted(_TRAINING_OPTIMIZERS))
            raise ValueError(
                f"RL does not map training optimizer {optimizer_name!r}; "
                f"supported: {supported}, or provide optimizer_type explicitly"
            ) from exc

    optimizer_args = {
        "lr": float(getattr(training_args, "learning_rate")),
        "weight_decay": float(getattr(training_args, "weight_decay")),
        "betas": (
            float(getattr(training_args, "adam_beta1")),
            float(getattr(training_args, "adam_beta2")),
        ),
        "eps": float(getattr(training_args, "adam_epsilon")),
    }
    return build_optimizer(
        model,
        extra_models=extra_models,
        optimizer_type=optimizer_type,
        args=optimizer_args,
        trainable_only=trainable_only,
    )


def build_torch_module(
    module_type, *, args=None, device=None, checkpoint=None
):
    """Instantiate one configured torch module and place it on the runtime device."""

    import torch.nn as nn

    module = import_symbol(module_type)(**dict(args or {}))
    if not isinstance(module, nn.Module):
        raise TypeError("configured module must inherit torch.nn.Module")
    if device is not None:
        module = module.to(device)
    if checkpoint is not None:
        from rl.critic import BaseCritic

        if not isinstance(module, BaseCritic):
            raise TypeError("checkpoint-backed module must inherit BaseCritic")
        module.load_pretrained(checkpoint)
    return module


def build_trainer_adapter_from_components(
    policy_components: Mapping,
    *,
    adapter: str = "auto",
    policy_adapter=None,
    algorithm=None,
    optimizer=None,
    scheduler=None,
    step_fn=None,
    adapter_args=None,
):
    """Compose a trainer adapter with an RL algorithm.

    The policy's normal Trainer class is only reused when it explicitly exposes
    ``build_trainer_adapter``. Full SFT ``Trainer.train()`` loops are never invoked by
    the RL runner.
    """

    if not isinstance(policy_components, Mapping):
        raise TypeError("policy_components must be a mapping")
    try:
        policy_module = policy_components["policy_module"]
    except KeyError as exc:
        raise KeyError("policy_components must contain policy_module") from exc

    native_trainer_class = None
    paths = policy_components.get("paths")
    if paths is not None:
        if not isinstance(paths, Mapping):
            raise TypeError("policy_components paths must be a mapping")
        policy_path = paths.get("policy")
        if policy_path is not None:
            native_trainer_class = get_policy_trainer_class(policy_path)

    kwargs = dict(adapter_args or {})
    optional_dependencies = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "step_fn": step_fn,
    }
    for name, value in optional_dependencies.items():
        if value is None:
            continue
        if name in kwargs:
            raise ValueError(f"{name} was provided twice")
        kwargs[name] = value

    parameter_group_builder = getattr(
        algorithm,
        "optimizer_parameter_groups",
        None,
    )
    if callable(parameter_group_builder):
        parameter_groups = parameter_group_builder(policy_adapter)
        if parameter_groups:
            if step_fn is not None:
                raise ValueError("algorithm-owned optimizer groups do not accept step_fn")
            return build_grouped_trainer_adapter(
                optimizer,
                parameter_groups,
                scheduler=scheduler,
            )

    return build_trainer_adapter(
        policy_module,
        policy_components,
        adapter=adapter,
        policy_adapter=policy_adapter,
        native_trainer_class=native_trainer_class,
        **kwargs,
    )
