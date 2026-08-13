"""Shared observation and activation utilities for critic modules."""

from collections.abc import Sequence

import torch
import torch.nn as nn

from benchmark.base import MetaObs


def observations(value):
    if isinstance(value, MetaObs):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if result and all(isinstance(item, MetaObs) for item in result):
            return result
    raise TypeError("critic input must be MetaObs or a non-empty MetaObs sequence")


def states(items, *, device):
    values = []
    for obs in items:
        state = obs.state
        if state is None:
            state = obs.state_ee if obs.state_ee is not None else obs.state_joint
        if state is None:
            raise ValueError("critic observation must contain state")
        values.append(
            torch.as_tensor(state, dtype=torch.float32, device=device).reshape(-1)
        )
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("critic state dimensions must agree within a batch")
    return torch.stack(values)


def activation(name):
    try:
        return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]
    except KeyError as exc:
        raise ValueError("activation must be relu, gelu, or tanh") from exc


__all__ = ["activation", "observations", "states"]
