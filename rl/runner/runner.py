"""Outer RL loop composing collection, algorithms and policy updates."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from data_utils.utils import set_seed
from rl.algorithm import AlgorithmUpdateResult, BaseRLAlgorithm
from rl.buffer import BaseBuffer, RolloutBuffer
from rl.collector import BaseCollector, CollectResult
from rl.policy_adapter import BasePolicyAdapter
from rl.policy_adapter.trainer import BaseTrainerAdapter

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
        collector: BaseCollector,
        buffer: BaseBuffer,
        policy_adapter: BasePolicyAdapter,
        algorithm: BaseRLAlgorithm,
        trainer_adapter: BaseTrainerAdapter,
        config: RLRunnerConfig,
        callbacks=(),
    ) -> None:
        if not isinstance(collector, BaseCollector):
            raise TypeError("collector must inherit BaseCollector")
        if not isinstance(buffer, BaseBuffer):
            raise TypeError("buffer must inherit BaseBuffer")
        if getattr(collector, "buffer", None) is not buffer:
            raise ValueError("collector and RLRunner must share the same buffer")
        if not isinstance(policy_adapter, BasePolicyAdapter):
            raise TypeError("policy_adapter must inherit BasePolicyAdapter")
        if not isinstance(algorithm, BaseRLAlgorithm):
            raise TypeError("algorithm must inherit BaseRLAlgorithm")
        if not isinstance(trainer_adapter, BaseTrainerAdapter):
            raise TypeError("trainer_adapter must inherit BaseTrainerAdapter")
        if not isinstance(config, RLRunnerConfig):
            raise TypeError("config must be RLRunnerConfig")
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
        kwargs = {
            "deterministic": self.config.deterministic_collection,
            "context": context,
        }
        if self.config.collect_steps is not None:
            kwargs["num_steps"] = self.config.collect_steps
        else:
            kwargs["num_episodes"] = self.config.collect_episodes
        with self.policy_adapter.collection_context():
            return self.collector.collect(**kwargs)

    def _can_update(self) -> bool:
        if self.config.updates_per_iteration == 0 or len(self.buffer) == 0:
            return False
        return self.buffer.num_env_steps >= self.config.warmup_steps

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
            collection = self._collect(iteration_context)
            collection_seconds = perf_counter() - collection_started
            self.global_env_steps += collection.num_steps

            if isinstance(self.buffer, RolloutBuffer):
                self.buffer.seal()

            updates = []
            policy_was_updated = False
            update_started = perf_counter()
            if self._can_update():
                self.policy_adapter.set_training(True)
                for batch in self.algorithm.iter_update_batches(
                    self.buffer,
                    batch_size=self.config.batch_size,
                    num_updates=self.config.updates_per_iteration,
                    rng=self._rng,
                ):
                    update_context = self._context(context)
                    update = self.algorithm.update(
                        batch,
                        policy_adapter=self.policy_adapter,
                        trainer_adapter=self.trainer_adapter,
                        context=update_context,
                    )
                    updates.append(update)
                    if update.updated:
                        policy_was_updated = True
                        self.global_update_steps += 1
                        self.policy_adapter.bump_policy_version()
                self.policy_adapter.set_training(False)
            update_seconds = perf_counter() - update_started
            if policy_was_updated:
                self.collector.policy_updated()

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

    def state_dict(self):
        reward_composer = getattr(self.collector, "reward_composer", None)
        return {
            "version": self.STATE_VERSION,
            "iteration": self.iteration,
            "global_env_steps": self.global_env_steps,
            "global_update_steps": self.global_update_steps,
            "rng_state": self._rng.bit_generator.state,
            "algorithm": self.algorithm.state_dict(),
            "policy_adapter": self.policy_adapter.state_dict(),
            "trainer_adapter": self.trainer_adapter.state_dict(),
            "buffer": self.buffer.state_dict(),
            "reward_composer": (
                None if reward_composer is None else reward_composer.state_dict()
            ),
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("RLRunner state must be a mapping")
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

    def close(self):
        self.collector.close()
