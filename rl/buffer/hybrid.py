"""Replay buffer that mixes immutable offline data with online collection."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from rl.base import DecisionTransition, MetaTransition
from data_utils.offline_rl import RL_DATA_SOURCE_KEY

from .base import BaseBuffer, BufferBatch, TransitionStorage
from .offline import OfflineReplayBuffer


def _mark_online(item):
    def mark_transition(transition):
        info = dict(transition.info)
        info[RL_DATA_SOURCE_KEY] = "online"
        return replace(transition, info=info)

    if isinstance(item, MetaTransition):
        return mark_transition(item)
    if isinstance(item, DecisionTransition):
        steps = tuple(
            replace(step, transition=mark_transition(step.transition))
            for step in item.steps
        )
        extras = dict(item.decision.extras)
        extras[RL_DATA_SOURCE_KEY] = "online"
        return replace(
            item,
            decision=replace(item.decision, extras=extras),
            steps=steps,
        )
    raise TypeError("hybrid replay entries must be RL transitions")


class HybridReplayBuffer(BaseBuffer):
    """Route collection to online replay and sample a controlled data mixture."""

    STATE_VERSION = 1

    def __init__(
        self,
        offline_buffer: OfflineReplayBuffer,
        online_buffer: BaseBuffer,
        *,
        offline_ratio=0.5,
        seed=None,
    ):
        if not isinstance(offline_buffer, OfflineReplayBuffer):
            raise TypeError("hybrid offline_buffer must be OfflineReplayBuffer")
        if not isinstance(online_buffer, BaseBuffer):
            raise TypeError("hybrid online_buffer must inherit BaseBuffer")
        if offline_buffer.buffer_type != "replay" or online_buffer.buffer_type != "replay":
            raise ValueError("hybrid buffers must both provide replay semantics")
        if isinstance(offline_ratio, bool) or not isinstance(
            offline_ratio, (int, float)
        ):
            raise TypeError("offline_ratio must be a real number")
        if not 0.0 <= float(offline_ratio) <= 1.0:
            raise ValueError("offline_ratio must be in [0, 1]")
        super().__init__(TransitionStorage())
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer
        self.offline_ratio = float(offline_ratio)
        self._rng = np.random.default_rng(seed)
        self._marked_decisions = {}

    @property
    def buffer_type(self):
        return "replay"

    @property
    def capacity(self):
        online_capacity = self.online_buffer.capacity
        return None if online_capacity is None else len(self.offline_buffer) + online_capacity

    @property
    def num_env_steps(self):
        return self.offline_buffer.num_env_steps + self.online_buffer.num_env_steps

    @property
    def transitions(self):
        return self.offline_buffer.transitions + self.online_buffer.transitions

    def __len__(self):
        return len(self.offline_buffer) + len(self.online_buffer)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < len(self.offline_buffer):
            return self.offline_buffer[index]
        return self.online_buffer[index - len(self.offline_buffer)]

    def add(self, transition):
        return self.online_buffer.add(_mark_online(transition))

    def extend(self, transitions):
        return self.online_buffer.extend(tuple(_mark_online(item) for item in transitions))

    def add_step(self, step, decision):
        add_step = getattr(self.online_buffer, "add_step", None)
        if callable(add_step):
            pending = self._marked_decisions.get(decision.decision_id)
            if pending is None:
                extras = dict(decision.extras)
                extras[RL_DATA_SOURCE_KEY] = "online"
                marked_decision = replace(decision, extras=extras)
                self._marked_decisions[decision.decision_id] = (
                    decision,
                    marked_decision,
                )
            else:
                original_decision, marked_decision = pending
                if original_decision is not decision:
                    raise ValueError("one decision_id refers to different decisions")
            marked_step = replace(
                step,
                transition=_mark_online(step.transition),
            )
            result = add_step(marked_step, marked_decision)
            if (
                step.transition.episode_done
                or step.action_offset + 1 == decision.chunk_size
            ):
                self._marked_decisions.pop(decision.decision_id, None)
            return result
        return self.add(step.transition)

    def flush_pending(self):
        flush = getattr(self.online_buffer, "flush_pending", None)
        result = flush() if callable(flush) else None
        self._marked_decisions.clear()
        return result

    def clear(self):
        self.online_buffer.clear()
        self._marked_decisions.clear()

    @staticmethod
    def _sample_counts(batch_size, offline_size, online_size, offline_ratio):
        offline_count = min(int(round(batch_size * offline_ratio)), offline_size)
        online_count = min(batch_size - offline_count, online_size)
        missing = batch_size - offline_count - online_count
        if missing:
            extra_offline = min(missing, offline_size - offline_count)
            offline_count += extra_offline
            missing -= extra_offline
        if missing:
            extra_online = min(missing, online_size - online_count)
            online_count += extra_online
            missing -= extra_online
        if missing:
            raise ValueError("batch_size exceeds hybrid replay size")
        return offline_count, online_count

    def get_batch(self, indices):
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        return BufferBatch(
            transitions=tuple(self[int(index)] for index in indices),
            indices=indices,
        )

    def sample(self, batch_size, *, replace=False):
        if replace:
            raise ValueError("hybrid replay does not support replacement sampling")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        offline_count, online_count = self._sample_counts(
            batch_size,
            len(self.offline_buffer),
            len(self.online_buffer),
            self.offline_ratio,
        )
        transitions = []
        indices = []
        if offline_count:
            batch = self.offline_buffer.sample(offline_count, replace=False)
            transitions.extend(batch.transitions)
            indices.extend(int(index) for index in batch.indices)
        if online_count:
            batch = self.online_buffer.sample(online_count, replace=False)
            transitions.extend(batch.transitions)
            indices.extend(
                len(self.offline_buffer) + int(index) for index in batch.indices
            )
        order = self._rng.permutation(batch_size)
        return BufferBatch(
            transitions=tuple(transitions[int(index)] for index in order),
            indices=np.asarray([indices[int(index)] for index in order], dtype=np.int64),
            metadata={
                "offline_samples": offline_count,
                "online_samples": online_count,
                "offline_fraction": offline_count / batch_size,
            },
        )

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "buffer_type": self.buffer_type,
            "offline_ratio": self.offline_ratio,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "offline_buffer": self.offline_buffer.state_dict(),
            "online_buffer": self.online_buffer.state_dict(),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported hybrid replay state")
        if state.get("buffer_type") != self.buffer_type:
            raise ValueError("hybrid replay buffer type mismatch")
        if float(state.get("offline_ratio")) != self.offline_ratio:
            raise ValueError("hybrid replay offline_ratio mismatch")
        self.offline_buffer.load_state_dict(state["offline_buffer"])
        self.online_buffer.load_state_dict(state["online_buffer"])
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self._marked_decisions.clear()


__all__ = ["HybridReplayBuffer"]
