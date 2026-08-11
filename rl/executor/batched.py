"""Batched in-process policy execution for parallel environment collection."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from benchmark.base import MetaObs
from deploy.action_manager.chunk import BasicActionChunkManager
from rl.base import RL_LIKELIHOOD_GROUP_KEY, RL_LIKELIHOOD_GROUP_SIZE_KEY
from rl.policy_adapter import MetaPolicyAdapter

from .base import BasePolicyExecutor
from .rl import RLPolicyExecutor


class BatchedRLPolicyExecutor(BasePolicyExecutor):
    """Batch policy inference while retaining one chunk manager per env lane."""

    def __init__(
        self,
        adapter: MetaPolicyAdapter,
        *,
        num_envs: int,
        execution_horizon: Optional[int] = None,
    ) -> None:
        if not isinstance(adapter, MetaPolicyAdapter):
            raise TypeError("adapter must inherit MetaPolicyAdapter")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.adapter = adapter
        self.num_envs = num_envs
        self._lanes = tuple(
            RLPolicyExecutor(
                adapter,
                chunk_manager=BasicActionChunkManager(),
                execution_horizon=execution_horizon,
                owns_adapter=False,
            )
            for _ in range(num_envs)
        )
        self._next_decision_id = 0
        self._next_likelihood_group_id = 0
        self._closed = False

    @property
    def inference_count(self) -> int:
        return sum(lane.inference_count for lane in self._lanes)

    def _indices(self, env_indices, count):
        if env_indices is None:
            env_indices = tuple(range(self.num_envs))
        else:
            env_indices = tuple(env_indices)
        if len(env_indices) != count:
            raise ValueError("env_indices and observations must have equal length")
        if len(set(env_indices)) != len(env_indices):
            raise ValueError("env_indices must be unique")
        for index in env_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("environment indices must be integers")
            if not 0 <= index < self.num_envs:
                raise IndexError("environment index is out of range")
        return env_indices

    def select_actions(
        self,
        observations: Sequence[MetaObs],
        *,
        env_indices=None,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
        if self._closed:
            raise RuntimeError("policy executor is closed")
        observations = tuple(observations)
        if not observations:
            raise ValueError("observations cannot be empty")
        if not all(isinstance(obs, MetaObs) for obs in observations):
            raise TypeError("observations must contain MetaObs values")
        env_indices = self._indices(env_indices, len(observations))

        inference_positions = tuple(
            position
            for position, env_index in enumerate(env_indices)
            if self._lanes[env_index].needs_inference
        )
        if inference_positions:
            inference_observations = tuple(
                observations[position] for position in inference_positions
            )
            outputs = tuple(
                self.adapter.select_actions(
                    inference_observations,
                    deterministic=deterministic,
                    context=context,
                )
            )
            if len(outputs) != len(inference_positions):
                raise ValueError("batched policy output size does not match inputs")
            likelihood_group_id = self._next_likelihood_group_id
            likelihood_group_size = len(outputs)
            for position, output in zip(inference_positions, outputs):
                env_index = env_indices[position]
                self._lanes[env_index].enqueue_policy_output(
                    output,
                    observations[position],
                    policy_info={
                        "decision_id": self._next_decision_id,
                        "env_index": env_index,
                        RL_LIKELIHOOD_GROUP_KEY: likelihood_group_id,
                        RL_LIKELIHOOD_GROUP_SIZE_KEY: likelihood_group_size,
                    },
                )
                self._next_decision_id += 1
            self._next_likelihood_group_id += 1

        return tuple(self._lanes[index].take_action() for index in env_indices)

    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ):
        if self.num_envs != 1:
            raise RuntimeError("use select_actions() for a multi-environment executor")
        return self.select_actions(
            (obs,),
            env_indices=(0,),
            deterministic=deterministic,
            context=context,
        )[0]

    def reset(self, env_indices=None) -> None:
        if self._closed:
            return
        if env_indices is None:
            env_indices = tuple(range(self.num_envs))
            reset_adapter = True
        else:
            env_indices = tuple(env_indices)
            env_indices = self._indices(env_indices, len(env_indices))
            reset_adapter = len(env_indices) == self.num_envs
        for index in env_indices:
            self._lanes[index].reset()
        if reset_adapter:
            reset = getattr(self.adapter, "reset", None)
            if callable(reset):
                reset()

    def policy_updated(self) -> None:
        if self._closed:
            return
        for lane in self._lanes:
            lane.policy_updated()
        hook = getattr(self.adapter, "policy_updated", None)
        if callable(hook):
            hook()

    def close(self) -> None:
        if self._closed:
            return
        try:
            for lane in self._lanes:
                lane.close()
            close = getattr(self.adapter, "close", None)
            if callable(close):
                close()
        finally:
            self._closed = True
