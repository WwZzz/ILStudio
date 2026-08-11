"""Train an ILStudio checkpoint with the modular RL pipeline."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from deploy.comm import is_server_address
from loguru import logger

from configs.loader import ConfigLoader
from data_utils.utils import set_seed
from rl.pipeline import build_rl_pipeline, compose_rl_config, validate_rl_config
from utils.torch_backend import configure_torch_backends_from_env

def summarize_rl_iteration(result, runner):
    episodes = result.collection.episodes
    collection_seconds = float(getattr(result, "collection_seconds", 0.0))
    update_seconds = float(getattr(result, "update_seconds", 0.0))
    metrics = {
        "iteration": result.iteration,
        "global_env_steps": runner.global_env_steps,
        "global_update_steps": runner.global_update_steps,
        "buffer_size": len(runner.buffer),
        "buffer_env_steps": runner.buffer.num_env_steps,
        "collected_steps": result.collection.num_steps,
        "collected_episodes": result.collection.num_episodes,
        "updates": len(result.updates),
        "runtime/collection_seconds": collection_seconds,
        "runtime/update_seconds": update_seconds,
        "runtime/iteration_seconds": collection_seconds + update_seconds,
    }
    if collection_seconds > 0:
        metrics["throughput/env_steps_per_second"] = (
            result.collection.num_steps / collection_seconds
        )
    for key, value in getattr(result.collection, "metrics", {}).items():
        if key in metrics:
            raise KeyError(f"collection metric conflicts with runner metric: {key}")
        metrics[key] = value
    if update_seconds > 0 and result.updates:
        metrics["throughput/updates_per_second"] = (
            len(result.updates) / update_seconds
        )
        objective_count = sum(
            float(value)
            for update in result.updates
            for key, value in update.metrics.items()
            if key.endswith("/num_objectives")
        )
        if objective_count:
            metrics["throughput/objectives_per_second"] = (
                objective_count / update_seconds
            )
    if episodes:
        metrics["episode/success_rate"] = fmean(
            float(episode.success) for episode in episodes
        )
        metrics["episode/length_mean"] = fmean(
            episode.length for episode in episodes
        )
        reward_keys = sorted(
            {key for episode in episodes for key in episode.reward}
        )
        for key in reward_keys:
            values = [episode.reward[key] for episode in episodes if key in episode.reward]
            metrics[f"episode_reward/{key}"] = fmean(float(value) for value in values)
    update_keys = sorted(
        {key for update in result.updates for key in update.metrics}
    )
    for key in update_keys:
        values = [update.metrics[key] for update in result.updates if key in update.metrics]
        metrics[key] = fmean(float(value) for value in values)
    return metrics


def build_metrics_callback(output_dir):
    path = Path(output_dir) / "rl_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def callback(result, runner):
        metrics = summarize_rl_iteration(result, runner)
        try:
            import torch

            if torch.cuda.is_available():
                metrics["gpu/peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )
                metrics["gpu/peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved() / 1024**3
                )
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        logger.info(
            "RL iteration={} env_steps={} updates={} success_rate={:.3f}",
            result.iteration,
            runner.global_env_steps,
            runner.global_update_steps,
            metrics.get("episode/success_rate", float("nan")),
        )

    return callback


def parse_param(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an ILStudio policy checkpoint with RL",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        help="Local ILStudio checkpoint used as the starting policy",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "hybrid"),
        default="online",
        help="Experience mode: environment, fixed task dataset, or both",
    )
    parser.add_argument(
        "-t",
        "--task",
        default=None,
        help="Task dataset config required by offline and hybrid modes",
    )
    parser.add_argument(
        "--offline-ratio",
        type=float,
        default=0.5,
        help="Offline fraction of each hybrid replay batch",
    )
    parser.add_argument(
        "--offline-pretrain-iterations",
        type=int,
        default=0,
        help="Hybrid update-only iterations before environment collection starts",
    )
    parser.add_argument(
        "--offline-item-type",
        choices=("auto", "transition", "decision"),
        default="auto",
        help="Replay item granularity; auto follows the algorithm buffer",
    )
    parser.add_argument(
        "--offline-missing-success",
        choices=("success", "failure"),
        default="success",
        help="Episode label used when an offline dataset omits success",
    )
    parser.add_argument(
        "-e",
        "--env",
        default="metaworld.easy00",
        help="Env config name under configs/env or an absolute YAML path",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="ppo",
        help="Algorithm config under configs/rl/algorithm",
    )
    parser.add_argument(
        "-r",
        "--reward",
        action="append",
        default=None,
        help=(
            "Reward config under configs/rl/reward; repeat to compose rewards "
            "(default: raw)"
        ),
    )
    parser.add_argument(
        "--env_runner",
        default="sync",
        help="Env-runner config under configs/rl/env_runner",
    )
    parser.add_argument(
        "--runner",
        default="default",
        help="Outer-loop config under configs/rl/runner",
    )
    parser.add_argument(
        "--objective",
        default=None,
        help=(
            "Policy objective config under configs/rl/objective; defaults to "
            "the algorithm config selection"
        ),
    )
    parser.add_argument(
        "--critic",
        default=None,
        help=(
            "Critic config under configs/rl/critic; defaults to the algorithm "
            "config selection"
        ),
    )
    parser.add_argument(
        "-c",
        "--training_config",
        default="rl",
        help="Optimizer/runtime config under configs/training",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--output-dir",
        default="ckpt/rl",
        help="Output ILStudio checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used to load and update the checkpoint",
    )
    parser.add_argument(
        "--env_index",
        type=int,
        default=None,
        help="Select one env from a multi-env config",
    )
    parser.add_argument(
        "--env_indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Select an explicit environment suite for grouped multi-task "
            "collection; mutually exclusive with --env_index"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configs and imports without loading model or environment",
    )
    args, unknown = parser.parse_known_args(argv)
    args.unknown_args = unknown
    args.reward = args.reward or ["raw"]
    if args.mode in {"offline", "hybrid"} and not args.task:
        parser.error("--task is required in offline and hybrid modes")
    if not 0.0 <= args.offline_ratio <= 1.0:
        parser.error("--offline-ratio must be in [0, 1]")
    if args.offline_pretrain_iterations < 0:
        parser.error("--offline-pretrain-iterations cannot be negative")
    if args.mode != "hybrid" and args.offline_pretrain_iterations:
        parser.error("--offline-pretrain-iterations requires --mode hybrid")
    if args.env_index is not None and args.env_indices is not None:
        parser.error("--env_index and --env_indices are mutually exclusive")
    return args


def _validate_checkpoint_source(model_name_or_path):
    if is_server_address(model_name_or_path):
        raise ValueError(
            "RL training requires a local checkpoint; remote policy servers "
            "cannot provide trainable parameters"
        )
    if model_name_or_path.startswith("__dummy-"):
        raise ValueError("RL training requires a real local policy checkpoint")


def load_all_configs(args):
    """Load every selected config through one native ILStudio ConfigLoader."""

    _validate_checkpoint_source(args.model_name_or_path)
    loader = ConfigLoader(
        args=args,
        unknown_args=getattr(args, "unknown_args", []),
    )
    uses_online = args.mode in {"online", "hybrid"}
    uses_offline = args.mode in {"offline", "hybrid"}
    env_config = None
    env_path = None
    if uses_online:
        env_config, env_path = loader.load_env(args.env)
    task_config = None
    task_path = None
    if uses_offline:
        task_config, task_path = loader.load_task(args.task)
    algorithm_config, algorithm_path = loader.load_rl_component(
        "algorithm",
        args.algorithm,
    )
    objective_name = getattr(args, "objective", None) or algorithm_config.get(
        "objective"
    )
    critic_name = getattr(args, "critic", None) or algorithm_config.get("critic")
    objective_config = None
    objective_path = None
    if objective_name is not None:
        objective_config, objective_path = loader.load_rl_component(
            "objective", objective_name
        )
    critic_config = None
    critic_path = None
    if critic_name is not None:
        critic_config, critic_path = loader.load_rl_component(
            "critic", critic_name
        )
    reward_configs = []
    reward_paths = []
    if uses_online:
        for reward_name in args.reward:
            reward_config, reward_path = loader.load_rl_component(
                "reward",
                reward_name,
            )
            reward_configs.append(reward_config)
            reward_paths.append(reward_path)
    env_runner_config = None
    env_runner_path = None
    if uses_online:
        env_runner_config, env_runner_path = loader.load_rl_component(
            "env_runner",
            args.env_runner,
        )
    runner_config, runner_path = loader.load_rl_component(
        "runner",
        args.runner,
    )
    training_config, training_args, training_path = loader.load_training(
        args.training_config,
        hyper_args=args,
    )

    graph = compose_rl_config(
        model_name_or_path=args.model_name_or_path,
        env_config=env_config,
        training_args=training_args,
        algorithm_config=algorithm_config,
        objective_config=objective_config,
        critic_config=critic_config,
        reward_configs=reward_configs,
        env_runner_config=env_runner_config,
        runner_config=runner_config,
        env_index=args.env_index,
        env_indices=args.env_indices,
        runtime_args={"device": args.device},
        mode=args.mode,
        offline_config=(
            {
                "task_config": task_config,
                "default_success": args.offline_missing_success == "success",
                "offline_ratio": args.offline_ratio,
                "offline_pretrain_iterations": args.offline_pretrain_iterations,
                "item_type": args.offline_item_type,
            }
            if uses_offline
            else None
        ),
    )
    paths = {
        "env": env_path,
        "task": task_path,
        "algorithm": algorithm_path,
        "objective": objective_path,
        "critic": critic_path,
        "reward": reward_paths,
        "env_runner": env_runner_path,
        "runner": runner_path,
        "training": training_path,
    }
    return graph, paths


def main(argv=None):
    args = argv if isinstance(argv, argparse.Namespace) else parse_param(argv)
    args.is_training = True
    config, config_paths = load_all_configs(args)
    if args.validate_only:
        validate_rl_config(config, import_targets=True)
        logger.info(f"RL configs are valid: {config_paths}")
        return None

    if configure_torch_backends_from_env():
        logger.info("cuDNN disabled by ILSTUDIO_DISABLE_CUDNN")

    seed = config.get("metadata", {}).get("seed")
    seed = 0 if seed is None else seed
    set_seed(seed)
    logger.info(f"Set global seed to: {seed} for reproducibility")
    logger.info(
        f"Starting {args.algorithm} RL from checkpoint: "
        f"{args.model_name_or_path}"
    )
    pipeline = build_rl_pipeline(config)
    try:
        try:
            import torch

            if torch.cuda.is_available():
                # Exclude checkpoint loading from per-iteration training peaks.
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        pipeline.entry.callbacks = (
            *tuple(getattr(pipeline.entry, "callbacks", ())),
            build_metrics_callback(args.output_dir),
        )
        result = pipeline.run()
        save_policy = getattr(pipeline.entry, "save_policy", None)
        if not callable(save_policy):
            raise TypeError("RLRunner must provide save_policy()")
        save_policy(args.output_dir)
        logger.info(f"Saved RL policy to: {args.output_dir}")
        return result
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
