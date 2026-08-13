"""Trajectory reward-weighted regression for chunk-action policies."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from rl.base import MetaTransition
from rl.buffer import BaseBuffer, RolloutBuffer
from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm


@dataclass(frozen=True)
class ChunkTrainingBatch:
    """Policy-native action chunks sampled from complete rollout episodes."""

    chunks: Tuple[Tuple[MetaTransition, ...], ...]
    trajectory_indices: Tuple[int, ...]
    trajectory_returns: Tuple[float, ...]
    trajectory_successes: Tuple[bool, ...]
    sampling_probabilities: Tuple[float, ...]

    def __post_init__(self):
        chunks = tuple(tuple(chunk) for chunk in self.chunks)
        if not chunks or any(not chunk for chunk in chunks):
            raise ValueError("chunk training batches must contain non-empty chunks")
        if any(
            not isinstance(transition, MetaTransition)
            for chunk in chunks
            for transition in chunk
        ):
            raise TypeError("training chunks must contain MetaTransition values")
        fields = (
            tuple(self.trajectory_indices),
            tuple(self.trajectory_returns),
            tuple(self.trajectory_successes),
            tuple(self.sampling_probabilities),
        )
        if any(len(field) != len(chunks) for field in fields):
            raise ValueError("chunk batch metadata must align with chunks")
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "trajectory_indices", fields[0])
        object.__setattr__(self, "trajectory_returns", fields[1])
        object.__setattr__(self, "trajectory_successes", fields[2])
        object.__setattr__(self, "sampling_probabilities", fields[3])

    def __len__(self):
        return len(self.chunks)


@dataclass(frozen=True)
class _Trajectory:
    index: int
    transitions: Tuple[MetaTransition, ...]
    chunks: Tuple[Tuple[MetaTransition, ...], ...]
    episode_return: float
    success: bool


def _scalar_reward(value, *, key):
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise TypeError(f"reward component {key!r} must be scalar per transition")
        result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"reward component {key!r} must be finite")
    return result


def _trajectory_chunks(transitions):
    chunks = []
    current = []
    current_id = None

    def flush():
        nonlocal current
        if current:
            chunks.append(tuple(current))
            current = []

    for transition in transitions:
        info = transition.policy_info
        if "chunk_id" not in info:
            flush()
            chunks.append((transition,))
            current_id = None
            continue

        chunk_id = info["chunk_id"]
        chunk_index = info.get("chunk_index")
        if chunk_index is not None and (
            isinstance(chunk_index, bool) or not isinstance(chunk_index, int)
        ):
            raise TypeError("chunk_index must be an integer when present")

        if not current or chunk_id != current_id:
            flush()
            if chunk_index not in {None, 0}:
                raise ValueError("a new action chunk must begin at chunk_index 0")
            current_id = chunk_id
        elif chunk_index is not None and chunk_index != len(current):
            raise ValueError("chunk_index values must be contiguous within a chunk")
        current.append(transition)

    flush()
    return tuple(chunks)


class RewardWeightedRegressionAlgorithm(BaseRLAlgorithm):
    """Regress policy actions on high-return trajectories from its own rollout.

    Reward weighting is implemented through elite filtering and stochastic chunk
    sampling. The policy adapter remains responsible for its native supervised loss,
    while the trainer adapter remains responsible only for parameter updates.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        reward_key: str = "train/total",
        elite_fraction: float = 1.0,
        temperature: Optional[float] = None,
        success_only: bool = False,
        fallback_to_best: bool = True,
    ):
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if (
            isinstance(elite_fraction, bool)
            or not isinstance(elite_fraction, Real)
            or not 0.0 < float(elite_fraction) <= 1.0
        ):
            raise ValueError("elite_fraction must be in (0, 1]")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise ValueError("temperature must be a positive finite number or None")
        if not isinstance(success_only, bool) or not isinstance(fallback_to_best, bool):
            raise TypeError("success_only and fallback_to_best must be bool values")
        super().__init__(
            required_capabilities=("action", "training_forward"),
            required_buffer_type="rollout",
        )
        self.reward_key = reward_key
        self.elite_fraction = float(elite_fraction)
        self.temperature = None if temperature is None else float(temperature)
        self.success_only = success_only
        self.fallback_to_best = fallback_to_best

    def _trajectories(self, buffer: RolloutBuffer):
        trajectories = []
        for index, (start, end) in enumerate(buffer.trajectory_ranges()):
            transitions = buffer.get_batch(
                range(start, end)
            ).transitions
            episode_return = 0.0
            for transition in transitions:
                try:
                    reward = transition.reward[self.reward_key]
                except KeyError as exc:
                    raise KeyError(
                        f"rollout transition is missing reward key {self.reward_key!r}"
                    ) from exc
                episode_return += _scalar_reward(reward, key=self.reward_key)
            trajectories.append(
                _Trajectory(
                    index=index,
                    transitions=transitions,
                    chunks=_trajectory_chunks(transitions),
                    episode_return=episode_return,
                    success=any(transition.success for transition in transitions),
                )
            )
        return tuple(trajectories)

    def _elite_trajectories(self, trajectories):
        candidates = trajectories
        if self.success_only:
            successful = tuple(item for item in trajectories if item.success)
            if successful:
                candidates = successful
            elif not self.fallback_to_best:
                return ()
        if not candidates:
            return ()
        count = max(1, math.ceil(len(candidates) * self.elite_fraction))
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-item.episode_return, item.index),
            )[:count]
        )

    def _episode_probabilities(self, trajectories):
        if not trajectories:
            return np.empty(0, dtype=np.float64)
        if self.temperature is None:
            return np.full(len(trajectories), 1.0 / len(trajectories))
        returns = np.asarray(
            [trajectory.episode_return for trajectory in trajectories],
            dtype=np.float64,
        )
        logits = (returns - returns.max()) / self.temperature
        weights = np.exp(logits)
        return weights / weights.sum()

    def iter_update_batches(
        self,
        buffer: BaseBuffer,
        *,
        batch_size: int,
        num_updates: int,
        rng: np.random.Generator,
    ):
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(num_updates, bool)
            or not isinstance(num_updates, int)
            or num_updates < 0
        ):
            raise ValueError("num_updates must be a non-negative integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if not isinstance(buffer, RolloutBuffer):
            raise TypeError("reward-weighted regression requires RolloutBuffer")
        if not buffer.sealed:
            raise RuntimeError("rollout buffer must be sealed before RWR sampling")
        trajectories = self._elite_trajectories(self._trajectories(buffer))
        if not trajectories or num_updates == 0:
            return

        episode_probabilities = self._episode_probabilities(trajectories)
        records = []
        record_probabilities = []
        for trajectory, probability in zip(trajectories, episode_probabilities):
            chunk_probability = probability / len(trajectory.chunks)
            for chunk in trajectory.chunks:
                records.append((trajectory, chunk))
                record_probabilities.append(chunk_probability)
        record_probabilities = np.asarray(record_probabilities, dtype=np.float64)
        record_probabilities = np.maximum(record_probabilities, np.finfo(np.float64).tiny)
        record_probabilities /= record_probabilities.sum()

        size = min(batch_size, len(records))
        for _ in range(num_updates):
            indices = rng.choice(
                len(records),
                size=size,
                replace=False,
                p=record_probabilities,
            )
            selected = [records[int(index)] for index in indices]
            yield ChunkTrainingBatch(
                chunks=tuple(chunk for _, chunk in selected),
                trajectory_indices=tuple(item.index for item, _ in selected),
                trajectory_returns=tuple(
                    item.episode_return for item, _ in selected
                ),
                trajectory_successes=tuple(item.success for item, _ in selected),
                sampling_probabilities=tuple(
                    float(record_probabilities[int(index)]) for index in indices
                ),
            )

    def compute_update(
        self,
        batch,
        *,
        policy_adapter: MetaPolicyAdapter,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AlgorithmOutput:
        if not isinstance(batch, ChunkTrainingBatch):
            raise TypeError("RWR updates require ChunkTrainingBatch")
        result = policy_adapter.training_forward(batch, context=context)
        if not isinstance(result, Mapping) or "loss" not in result:
            raise TypeError("policy training_forward must return a mapping containing loss")
        loss = result["loss"]
        if loss is None:
            raise ValueError("policy training_forward returned an empty RWR loss")
        loss_metric = loss.detach() if hasattr(loss, "detach") else loss
        metrics = {
            "policy/loss": loss_metric,
            "rwr/batch_chunks": len(batch),
            "rwr/batch_actions": sum(len(chunk) for chunk in batch.chunks),
            "rwr/return_mean": float(np.mean(batch.trajectory_returns)),
            "rwr/success_fraction": float(np.mean(batch.trajectory_successes)),
            "rwr/unique_trajectories": len(set(batch.trajectory_indices)),
        }
        for key, value in result.items():
            if key != "loss" and key not in {"num_chunks", "num_actions"}:
                metrics[f"policy/{key}"] = value
        return AlgorithmOutput(
            loss=loss,
            metrics=metrics,
            payload={"policy_output": dict(result)},
        )

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "reward_key": self.reward_key,
            "elite_fraction": self.elite_fraction,
            "temperature": self.temperature,
            "success_only": self.success_only,
            "fallback_to_best": self.fallback_to_best,
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("algorithm state must be a mapping")
        if state != self.state_dict():
            raise ValueError("RWR algorithm state does not match configured sampling semantics")
