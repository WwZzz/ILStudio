"""Convert ILStudio task datasets into lazy offline-RL replay items.

The supervised-training sample contract stays unchanged.  This module adds the
episode and transition semantics that RL needs at the point where a task
dataset is attached to an RL pipeline.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from benchmark.base import MetaAction, MetaObs
from rl.base import (
    DecisionTransition,
    MetaTransition,
    PolicyDecision,
    RolloutStep,
)


RL_DATA_SOURCE_KEY = "rl.data_source"
RL_SUCCESS_DEFAULTED_KEY = "rl.success_defaulted"


def _numpy(value):
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value)


def _scalar(value, *, name):
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        array = _numpy(value)
        if array.size != 1:
            raise TypeError(f"offline sample {name} must be scalar")
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"offline sample {name} must be finite")
    return result


def _boolean(value, *, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = _numpy(value)
    if array.size != 1:
        raise TypeError(f"offline sample {name} must be scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if scalar in (0, 1):
        return bool(scalar)
    raise TypeError(f"offline sample {name} must be boolean")


def _sample_value(sample, *names):
    for name in names:
        if name in sample:
            return sample[name]
    info = sample.get("info")
    if isinstance(info, Mapping):
        for name in names:
            if name in info:
                return info[name]
    return None


def _nested_attribute(value, name):
    """Find metadata through transparent dataset wrappers."""

    seen = set()
    current = value
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "dataset", None)
    return None


@dataclass(frozen=True)
class _DatasetLayout:
    dataset: object
    episode_ends: tuple[int, ...]
    length: int
    ctrl_space: str
    ctrl_type: str
    dataset_id: str

    def episode_bounds(self, index):
        episode = bisect_right(self.episode_ends, index)
        start = 0 if episode == 0 else self.episode_ends[episode - 1]
        return episode, start, self.episode_ends[episode]


class OfflineReplayDataset:
    """Lazy replay view over map-style ILStudio task datasets.

    Existing samples may optionally contain ``success``, ``reward``,
    ``terminated`` and ``truncated``.  When an episode has no success label,
    ``default_success=True`` treats the demonstration as successful and marks
    only its terminal transition as success.  Missing rewards become a sparse
    terminal success reward under ``reward_key``.
    """

    def __init__(
        self,
        datasets,
        *,
        item_type="transition",
        default_success=True,
        reward_key="train/total",
        step_reward=0.0,
        success_reward=1.0,
        failure_reward=0.0,
    ):
        datasets = tuple(datasets)
        if not datasets:
            raise ValueError("offline RL requires at least one dataset")
        if item_type not in {"transition", "decision"}:
            raise ValueError("offline item_type must be transition or decision")
        if not isinstance(default_success, bool):
            raise TypeError("offline default_success must be bool")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("offline reward_key must be a non-empty string")
        self.item_type = item_type
        self.default_success = default_success
        self.reward_key = reward_key
        self.step_reward = float(step_reward)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self._success_cache = {}
        if not all(
            np.isfinite(value)
            for value in (self.step_reward, self.success_reward, self.failure_reward)
        ):
            raise ValueError("offline default rewards must be finite")

        self.layouts = tuple(self._layout(dataset) for dataset in datasets)
        total = 0
        ends = []
        for layout in self.layouts:
            total += layout.length
            ends.append(total)
        self._dataset_ends = tuple(ends)

    @staticmethod
    def _episode_lengths(dataset):
        length = len(dataset)
        configured = _nested_attribute(dataset, "episode_len")
        if configured is None:
            getter = _nested_attribute(dataset, "get_episode_len")
            configured = getter() if callable(getter) else None
        if configured is not None:
            lengths = tuple(int(value) for value in configured)
            if lengths and all(value > 0 for value in lengths) and sum(lengths) == length:
                return lengths

        # Generic fallback for map datasets that only expose episode_id.
        lengths = []
        previous = object()
        current_length = 0
        for index in range(length):
            sample = dataset[index]
            if not isinstance(sample, Mapping):
                raise TypeError("offline dataset samples must be mappings")
            episode_id = sample.get("episode_id", 0)
            episode_array = _numpy(episode_id)
            if episode_array.size == 1:
                episode_id = episode_array.reshape(-1)[0].item()
            if current_length and episode_id != previous:
                lengths.append(current_length)
                current_length = 0
            previous = episode_id
            current_length += 1
        if current_length:
            lengths.append(current_length)
        return tuple(lengths)

    def _layout(self, dataset):
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("offline replay currently requires map-style datasets")
        length = int(len(dataset))
        if length <= 0:
            raise ValueError("offline datasets cannot be empty")
        episode_lengths = self._episode_lengths(dataset)
        episode_ends = tuple(int(value) for value in np.cumsum(episode_lengths))
        if not episode_ends or episode_ends[-1] != length:
            raise ValueError("offline episode lengths do not cover the dataset")

        ctrl_space = _nested_attribute(dataset, "ctrl_space")
        ctrl_type = _nested_attribute(dataset, "ctrl_type")
        if ctrl_space is None:
            ctrl_space = "ee"
        if ctrl_type is None:
            ctrl_type = "delta"
        dataset_id = _nested_attribute(dataset, "dataset_id")
        if dataset_id is None:
            dataset_id = _nested_attribute(dataset, "name")
        return _DatasetLayout(
            dataset=dataset,
            episode_ends=episode_ends,
            length=length,
            ctrl_space=str(ctrl_space),
            ctrl_type=str(ctrl_type),
            dataset_id=str(dataset_id if dataset_id is not None else "unknown"),
        )

    def __len__(self):
        return self._dataset_ends[-1]

    def _locate(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("offline replay index out of range")
        dataset_index = bisect_right(self._dataset_ends, index)
        dataset_start = 0 if dataset_index == 0 else self._dataset_ends[dataset_index - 1]
        local_index = index - dataset_start
        layout = self.layouts[dataset_index]
        episode, episode_start, episode_end = layout.episode_bounds(local_index)
        return dataset_index, layout, local_index, episode, episode_start, episode_end

    @staticmethod
    def _sample(layout, index):
        sample = layout.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("offline dataset samples must be mappings")
        return sample

    @staticmethod
    def _observation(sample, *, fallback_timestep):
        values = {}
        for name in (
            "state",
            "state_ee",
            "state_joint",
            "state_obj",
            "image",
            "depth",
            "pc",
        ):
            if name in sample and sample[name] is not None:
                values[name] = _numpy(sample[name]).copy()
        raw_lang = sample.get("raw_lang", sample.get("language_instruction", ""))
        if raw_lang is None:
            raw_lang = ""
        timestep = sample.get("timestamp", sample.get("timestep", fallback_timestep))
        timestep_array = _numpy(timestep)
        if timestep_array.size != 1:
            raise TypeError("offline sample timestamp must be scalar")
        values["raw_lang"] = str(raw_lang)
        values["timestep"] = int(timestep_array.reshape(-1)[0])
        return MetaObs(**values)

    @staticmethod
    def _action_chunk(sample):
        if "action" not in sample:
            raise KeyError("offline sample is missing action")
        action = _numpy(sample["action"]).astype(np.float32, copy=True)
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] <= 0:
            raise ValueError("offline action must have shape [chunk, action]")
        is_pad = sample.get("is_pad")
        if is_pad is None:
            valid = np.ones(action.shape[0], dtype=bool)
        else:
            is_pad = _numpy(is_pad).astype(bool, copy=False).reshape(-1)
            if len(is_pad) != action.shape[0]:
                raise ValueError("offline is_pad must match the action chunk")
            valid = ~is_pad
        valid_count = int(valid.sum())
        if valid_count <= 0 or not valid[:valid_count].all() or valid[valid_count:].any():
            raise ValueError("offline action padding must be a non-empty valid prefix")
        return action, valid_count

    def _episode_success(self, layout, episode_end):
        key = (id(layout.dataset), episode_end)
        cached = self._success_cache.get(key)
        if cached is not None:
            return cached
        terminal_sample = self._sample(layout, episode_end - 1)
        value = _sample_value(
            terminal_sample,
            "success",
            "is_success",
            "episode_success",
        )
        if value is None:
            result = (self.default_success, True)
        else:
            result = (_boolean(value, name="success"), False)
        self._success_cache[key] = result
        return result

    def _reward(self, sample, *, terminal, success):
        value = _sample_value(sample, "reward", "rewards")
        if value is None:
            if terminal:
                value = self.success_reward if success else self.failure_reward
            else:
                value = self.step_reward
            return {self.reward_key: float(value)}
        if isinstance(value, Mapping):
            return {
                str(key): _scalar(item, name=f"reward[{key!r}]")
                for key, item in value.items()
            }
        return {self.reward_key: _scalar(value, name="reward")}

    def _transition(
        self,
        layout,
        local_index,
        *,
        episode,
        episode_end,
        action_override=None,
    ):
        sample = self._sample(layout, local_index)
        terminal_position = local_index == episode_end - 1
        next_sample = (
            sample
            if terminal_position
            else self._sample(layout, local_index + 1)
        )
        episode_success, success_defaulted = self._episode_success(layout, episode_end)
        explicit_success = _sample_value(sample, "success", "is_success")
        step_success = (
            _boolean(explicit_success, name="success")
            if explicit_success is not None
            else bool(terminal_position and episode_success)
        )

        explicit_terminated = _sample_value(sample, "terminated")
        explicit_truncated = _sample_value(sample, "truncated")
        terminated = (
            _boolean(explicit_terminated, name="terminated")
            if explicit_terminated is not None
            else terminal_position
        )
        truncated = (
            _boolean(explicit_truncated, name="truncated")
            if explicit_truncated is not None
            else False
        )
        if terminal_position and not (terminated or truncated):
            terminated = True

        if action_override is None:
            action_override = self._action_chunk(sample)[0][0]
        info = dict(sample.get("info", {})) if isinstance(sample.get("info"), Mapping) else {}
        info.update(
            {
                RL_DATA_SOURCE_KEY: "offline",
                "dataset_id": sample.get("dataset_id", layout.dataset_id),
                "episode_id": sample.get("episode_id", episode),
                "success": step_success,
            }
        )
        if success_defaulted:
            info[RL_SUCCESS_DEFAULTED_KEY] = True
        return MetaTransition(
            obs=self._observation(sample, fallback_timestep=local_index),
            action=MetaAction(
                action=np.asarray(action_override, dtype=np.float32).copy(),
                ctrl_space=layout.ctrl_space,
                ctrl_type=layout.ctrl_type,
            ),
            next_obs=self._observation(
                next_sample,
                fallback_timestep=local_index + (not terminal_position),
            ),
            reward=self._reward(sample, terminal=terminal_position, success=episode_success),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _decision(self, dataset_index, layout, local_index, episode, episode_end):
        sample = self._sample(layout, local_index)
        action, valid_count = self._action_chunk(sample)
        duration = min(valid_count, episode_end - local_index)
        decision_id = ("offline", dataset_index, episode, local_index)
        extras = {
            RL_DATA_SOURCE_KEY: "offline",
            "action_mask": np.arange(action.shape[0]) < duration,
        }
        decision = PolicyDecision(
            decision_id=decision_id,
            obs=self._observation(sample, fallback_timestep=local_index),
            action=MetaAction(
                action=action,
                ctrl_space=layout.ctrl_space,
                ctrl_type=layout.ctrl_type,
            ),
            extras=extras,
        )
        steps = []
        for offset in range(duration):
            transition = self._transition(
                layout,
                local_index + offset,
                episode=episode,
                episode_end=episode_end,
                action_override=action[offset],
            )
            steps.append(
                RolloutStep(
                    transition=transition,
                    decision_id=decision_id,
                    action_offset=offset,
                )
            )
        return DecisionTransition(decision=decision, steps=tuple(steps))

    def __getitem__(self, index):
        (
            dataset_index,
            layout,
            local_index,
            episode,
            _episode_start,
            episode_end,
        ) = self._locate(index)
        if self.item_type == "decision":
            return self._decision(
                dataset_index,
                layout,
                local_index,
                episode,
                episode_end,
            )
        return self._transition(
            layout,
            local_index,
            episode=episode,
            episode_end=episode_end,
        )


__all__ = [
    "OfflineReplayDataset",
    "RL_DATA_SOURCE_KEY",
    "RL_SUCCESS_DEFAULTED_KEY",
]
