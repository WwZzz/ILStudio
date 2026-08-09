"""Tensor and transition helpers shared by RL algorithms."""

from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch

from rl.base import DecisionTransition
from rl.buffer import BufferBatch


def transitions(batch):
    if not isinstance(batch, BufferBatch):
        raise TypeError("algorithm updates require BufferBatch")
    if not batch.transitions:
        raise ValueError("algorithm batch must not be empty")
    return batch.transitions


def forward_result(policy_adapter, operation, batch, *, context=None, required=()):
    result = policy_adapter.algorithm_forward(operation, batch, context=context)
    if not isinstance(result, Mapping):
        raise TypeError("policy algorithm_forward must return a mapping")
    missing = set(required) - set(result)
    if missing:
        raise KeyError(
            f"{operation} policy output is missing: {', '.join(sorted(missing))}"
        )
    return result


def vector(value, *, name):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 0:
        value = value.reshape(1)
    elif value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 1:
        raise ValueError(f"{name} must contain one scalar per transition")
    return value


def _scalar(value, *, name):
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise TypeError(f"{name} must be scalar per transition")
        result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def transition_values(items, getter, *, like, name):
    values = [_scalar(getter(item), name=name) for item in items]
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def rewards(items, key, *, like):
    values = []
    for item in items:
        if key not in item.reward:
            raise KeyError(f"transition reward is missing key {key!r}")
        values.append(_scalar(item.reward[key], name=f"reward {key!r}"))
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def decision_rewards(items, key, gamma, *, like):
    """Return rewards discounted inside each SMDP decision."""

    values = []
    for item in items:
        if isinstance(item, DecisionTransition):
            values.append(item.discounted_reward(key, gamma))
        else:
            if key not in item.reward:
                raise KeyError(f"transition reward is missing key {key!r}")
            values.append(_scalar(item.reward[key], name=f"reward {key!r}"))
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def bootstrap_discounts(items, gamma, *, like):
    """Return ``gamma ** duration`` times the termination bootstrap mask."""

    return torch.as_tensor(
        [
            (float(gamma) ** (item.duration if isinstance(item, DecisionTransition) else 1))
            * float(item.bootstrap_mask)
            for item in items
        ],
        dtype=like.dtype,
        device=like.device,
    )


def bootstrap_masks(items, *, like):
    return torch.as_tensor(
        [float(item.bootstrap_mask) for item in items],
        dtype=like.dtype,
        device=like.device,
    )


def discrete_actions(items, *, attribute="action", device=None):
    values = []
    for item in items:
        raw = getattr(item, attribute).action if attribute == "action" else item.policy_info[attribute]
        array = np.asarray(raw)
        if array.size != 1:
            raise ValueError(f"{attribute} must contain one discrete action")
        value = float(array.reshape(-1)[0])
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{attribute} must be an integer action index")
        values.append(int(value))
    return torch.as_tensor(values, dtype=torch.long, device=device)


def q_matrix(value, *, name, batch_size):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [batch, num_actions]")
    return value


def discounted_returns(items, reward_values, gamma):
    result = torch.empty_like(reward_values)
    running = torch.zeros((), dtype=reward_values.dtype, device=reward_values.device)
    for index in range(len(items) - 1, -1, -1):
        if items[index].episode_done:
            running = torch.zeros_like(running)
        running = reward_values[index] + gamma * running
        result[index] = running
    return result


def normalized(value, *, epsilon=1e-8):
    if value.numel() <= 1:
        return value - value.mean()
    return (value - value.mean()) / (value.std(unbiased=False) + epsilon)


def detached_metric(value):
    return value.detach() if hasattr(value, "detach") else value
