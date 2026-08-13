"""Outer RL loop composing collection, algorithms and policy updates."""

from dataclasses import dataclass, replace
from numbers import Real
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from data_utils.utils import set_seed
from rl.algorithm import AlgorithmUpdateResult, BaseRLAlgorithm
from rl.buffer import BaseBuffer, RolloutBuffer
from rl.collector import BaseCollector, CollectResult
from rl.policy_adapter import MetaPolicyAdapter, TrainerAdapter

from .config import RLRunnerConfig


@dataclass(frozen=True)
class RLIterationResult:
    iteration: int
    collection: CollectResult
    updates: Tuple[AlgorithmUpdateResult, ...]
    collection_seconds: float = 0.0
    update_seconds: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "updates", tuple(self.updates))


class RLRunner:
    """Compose the RL lifecycle without replacing policy-specific trainers."""

    STATE_VERSION = 2

    def __init__(
        self,
        *,
        collector: Optional[BaseCollector] = None,
        buffer: BaseBuffer,
        policy_adapter: MetaPolicyAdapter,
        algorithm: BaseRLAlgorithm,
        trainer_adapter: TrainerAdapter,
        config: RLRunnerConfig,
        callbacks=(),
    ) -> None:
        if collector is not None and not isinstance(collector, BaseCollector):
            raise TypeError("collector must inherit BaseCollector or be None")
        if not isinstance(buffer, BaseBuffer):
            raise TypeError("buffer must inherit BaseBuffer")
        if collector is not None and getattr(collector, "buffer", None) is not buffer:
            raise ValueError("collector and RLRunner must share the same buffer")
        if not isinstance(policy_adapter, MetaPolicyAdapter):
            raise TypeError("policy_adapter must inherit MetaPolicyAdapter")
        if not isinstance(algorithm, BaseRLAlgorithm):
            raise TypeError("algorithm must inherit BaseRLAlgorithm")
        if not isinstance(trainer_adapter, TrainerAdapter):
            raise TypeError("trainer_adapter must inherit TrainerAdapter")
        if not isinstance(config, RLRunnerConfig):
            raise TypeError("config must be RLRunnerConfig")
        if config.mode == "offline":
            if collector is not None:
                raise ValueError("offline mode must not construct a collector")
            if buffer.buffer_type != "replay":
                raise ValueError("offline mode requires replay buffer semantics")
        elif collector is None:
            raise ValueError(f"{config.mode} mode requires a collector")
        if config.mode == "hybrid" and buffer.buffer_type != "replay":
            raise ValueError("hybrid mode requires replay buffer semantics")
        callbacks = tuple(callbacks)
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("RLRunner callbacks must be callable")

        algorithm.validate(policy_adapter, buffer)
        if config.seed is not None:
            set_seed(config.seed)
        self.collector = collector
        self.buffer = buffer
        self.policy_adapter = policy_adapter
        self.algorithm = algorithm
        self.trainer_adapter = trainer_adapter
        self.config = config
        self.callbacks = callbacks
        self._rng = np.random.default_rng(config.seed)
        self.iteration = 0
        self.global_env_steps = 0
        self.global_update_steps = 0

    def _context(self, context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        result = dict(context or {})
        result.update(
            {
                "iteration": self.iteration,
                "global_env_steps": self.global_env_steps,
                "global_update_steps": self.global_update_steps,
            }
        )
        return result

    def _collect(self, context):
        if self.collector is None:
            raise RuntimeError("offline mode has no collector")
        collection_context = dict(context)
        algorithm = getattr(self, "algorithm", None)
        algorithm_context = getattr(algorithm, "collection_context", None)
        if callable(algorithm_context):
            collection_context.update(algorithm_context(context))
        kwargs = {
            "deterministic": self.config.deterministic_collection,
            "context": collection_context,
        }
        if self.config.collect_steps is not None:
            kwargs["num_steps"] = self.config.collect_steps
        else:
            kwargs["num_episodes"] = self.config.collect_episodes
        with self.policy_adapter.collection_context():
            return self.collector.collect(**kwargs)

    def _collect_for_update(self, context):
        total_steps = 0
        rejected_steps = 0
        for attempt in range(1, self.config.max_collection_attempts + 1):
            collection = self._collect(context)
            total_steps += collection.num_steps
            acceptance = self.algorithm.evaluate_collection(collection)
            exhausted = attempt == self.config.max_collection_attempts
            if acceptance.accepted or exhausted:
                if exhausted and not acceptance.accepted:
                    self.buffer.clear()
                metrics = dict(collection.metrics)
                metrics.update(acceptance.metrics)
                metrics.update(
                    {
                        "collection/attempts": float(attempt),
                        "collection/rejected_steps": float(rejected_steps),
                        "collection/total_steps": float(total_steps),
                        "collection/filter_exhausted": float(
                            exhausted and not acceptance.accepted
                        ),
                    }
                )
                return replace(collection, metrics=metrics), total_steps
            rejected_steps += collection.num_steps
            self.buffer.clear()
        raise RuntimeError("collection attempt loop did not produce a rollout")

    def _can_update(self) -> bool:
        if self.config.updates_per_iteration == 0 or len(self.buffer) == 0:
            return False
        return self.buffer.num_env_steps >= self.config.warmup_steps

    def _iter_update_batches(self):
        return self.algorithm.iter_update_batches(
            self.buffer,
            batch_size=self.config.batch_size,
            num_updates=self.config.updates_per_iteration,
            rng=self._rng,
        )

    def _run_updates(self, context):
        updates = []
        policy_was_updated = False
        update_started = perf_counter()

        if not self._can_update():
            return tuple(updates), perf_counter() - update_started

        self.policy_adapter.set_training(True)
        try:
            for batch in self._iter_update_batches():
                update_context = self._context(context)
                update = self.algorithm.update(
                    batch,
                    policy_adapter=self.policy_adapter,
                    trainer_adapter=self.trainer_adapter,
                    context=update_context,
                )
                sampling_metrics = {
                    f"replay/{key}": float(value)
                    for key, value in getattr(batch, "metadata", {}).items()
                    if isinstance(value, Real) and not isinstance(value, bool)
                }
                collisions = set(update.metrics).intersection(sampling_metrics)
                if collisions:
                    raise KeyError(
                        "algorithm metrics conflict with replay metrics: "
                        + ", ".join(sorted(collisions))
                    )
                if sampling_metrics:
                    update = replace(
                        update,
                        metrics={**update.metrics, **sampling_metrics},
                    )
                updates.append(update)
                if update.updated:
                    policy_was_updated = True
                    self.global_update_steps += 1
                    self.policy_adapter.bump_policy_version()
        finally:
            self.policy_adapter.set_training(False)

        if policy_was_updated and self.collector is not None:
            self.collector.policy_updated()

        return tuple(updates), perf_counter() - update_started

    def _should_collect(self):
        if self.config.mode == "offline":
            return False
        if self.config.mode == "hybrid" and (
            self.iteration < self.config.offline_pretrain_iterations
        ):
            return False
        return True

    def _empty_collection(self):
        phase = (
            "offline_pretrain"
            if self.config.mode == "hybrid"
            else "offline"
        )
        return CollectResult(
            transitions=(),
            episodes=(),
            metrics={
                "collection/skipped": 1.0,
                f"collection/phase_{phase}": 1.0,
            },
        )

    def run(
        self,
        *,
        iterations: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
        retain_results: bool = True,
    ):
        iterations = self.config.iterations if iterations is None else iterations
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(retain_results, bool):
            raise TypeError("retain_results must be bool")

        results = []
        for _ in range(iterations):
            if isinstance(self.buffer, RolloutBuffer) and (
                len(self.buffer) > 0 or self.buffer.sealed
            ):
                raise RuntimeError("rollout buffer must be empty at iteration start")

            iteration_context = self._context(context)
            self.policy_adapter.set_training(False)
            collection_started = perf_counter()
            if self._should_collect():
                collection, attempted_env_steps = self._collect_for_update(
                    iteration_context
                )
            else:
                collection = self._empty_collection()
                attempted_env_steps = 0
            collection_seconds = perf_counter() - collection_started
            self.global_env_steps += attempted_env_steps

            if isinstance(self.buffer, RolloutBuffer):
                self.buffer.seal()

            updates, update_seconds = self._run_updates(context)

            if (
                isinstance(self.buffer, RolloutBuffer)
                and self.config.clear_rollout_after_update
            ):
                self.buffer.clear()

            result = RLIterationResult(
                iteration=self.iteration,
                collection=collection,
                updates=tuple(updates),
                collection_seconds=collection_seconds,
                update_seconds=update_seconds,
            )
            self.iteration += 1
            for callback in self.callbacks:
                callback(result, self)
            if retain_results:
                results.append(result)
        return tuple(results)

    def state_dict(self, *, include_buffer=True):
        if not isinstance(include_buffer, bool):
            raise TypeError("include_buffer must be bool")
        reward_composer = getattr(self.collector, "reward_composer", None)
        state = {
            "version": self.STATE_VERSION,
            "iteration": self.iteration,
            "global_env_steps": self.global_env_steps,
            "global_update_steps": self.global_update_steps,
            "rng_state": self._rng.bit_generator.state,
            "algorithm": self.algorithm.state_dict(),
            "policy_adapter": self.policy_adapter.state_dict(),
            "trainer_adapter": self.trainer_adapter.state_dict(),
            "reward_composer": (
                None if reward_composer is None else reward_composer.state_dict()
            ),
        }
        if include_buffer:
            state["buffer"] = self.buffer.state_dict()
        return state

    def load_state_dict(self, state, *, load_buffer=True):
        if not isinstance(state, Mapping):
            raise TypeError("RLRunner state must be a mapping")
        if not isinstance(load_buffer, bool):
            raise TypeError("load_buffer must be bool")
        version = state.get("version")
        if version not in {1, self.STATE_VERSION}:
            raise ValueError("unsupported RLRunner state version")
        for key in ("iteration", "global_env_steps", "global_update_steps"):
            value = state.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")

        self.algorithm.load_state_dict(state["algorithm"])
        self.policy_adapter.load_state_dict(state["policy_adapter"])
        trainer_state_key = (
            "policy_trainer" if version == 1 else "trainer_adapter"
        )
        self.trainer_adapter.load_state_dict(state[trainer_state_key])
        if load_buffer:
            self.buffer.load_state_dict(state["buffer"])
        reward_state = state.get("reward_composer")
        reward_composer = getattr(self.collector, "reward_composer", None)
        if reward_state is not None:
            if reward_composer is None:
                raise ValueError("state contains reward state but collector has none")
            reward_composer.load_state_dict(reward_state)
        self._rng.bit_generator.state = state["rng_state"]
        self.iteration = state["iteration"]
        self.global_env_steps = state["global_env_steps"]
        self.global_update_steps = state["global_update_steps"]

    def save_policy(self, output_dir):
        result = self.policy_adapter.save_pretrained(output_dir)
        critic = getattr(self.algorithm, "critic", None)
        if critic is not None:
            save_critic = getattr(critic, "save_pretrained", None)
            if not callable(save_critic):
                raise TypeError("composed critic must provide save_pretrained()")
            save_critic(output_dir)
        return result

    def save_checkpoint(self, output_dir, *, replay_path=None):
        from .checkpoint import save_rl_checkpoint

        return save_rl_checkpoint(self, output_dir, replay_path=replay_path)

    def load_checkpoint(self, path, *, replay_path=None):
        from .checkpoint import load_rl_checkpoint

        return load_rl_checkpoint(self, path, replay_path=replay_path)

    def load_replay_checkpoint(self, path):
        from .checkpoint import load_replay_checkpoint

        return load_replay_checkpoint(self.buffer, path)

    def close(self):
        if self.collector is not None:
            self.collector.close()
