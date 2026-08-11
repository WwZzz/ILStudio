"""Single-Q critic with an algorithm-updated target copy."""

import copy
import importlib

import torch
from torch.nn.parameter import UninitializedParameter

from ..base import BaseCritic


class TargetQCritic(BaseCritic):
    def __init__(self, *, q_type="rl.critic.StateActionQCritic", q_args=None):
        super().__init__()
        if not isinstance(q_type, str) or "." not in q_type:
            raise TypeError("q_type must be a fully-qualified import path")
        module_name, symbol_name = q_type.rsplit(".", 1)
        q_class = getattr(importlib.import_module(module_name), symbol_name)
        self.q = q_class(**dict(q_args or {}))
        if not isinstance(self.q, BaseCritic):
            raise TypeError("TargetQCritic q_type must construct BaseCritic")
        self.target_q = copy.deepcopy(self.q)
        self.target_q.eval()

    def parameters(self, recurse=True):
        yield from self.q.parameters(recurse=recurse)

    def train(self, mode=True):
        super().train(mode)
        self.target_q.eval()
        return self

    def forward(self, obs, action, *, action_mask=None, context=None):
        return self.q(obs, action, action_mask=action_mask, context=context)

    def target(self, obs, action, *, action_mask=None, context=None):
        if any(
            isinstance(parameter, UninitializedParameter)
            for parameter in self.target_q.parameters()
        ):
            self.target_q.load_state_dict(self.q.state_dict())
            self.target_q.requires_grad_(False)
        with torch.no_grad():
            return self.target_q(
                obs, action, action_mask=action_mask, context=context
            )

    def parameter_groups(self):
        return {"critic": tuple(self.q.parameters())}

    def soft_update(self, tau):
        tau = float(tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                self.target_q.parameters(), self.q.parameters()
            ):
                target_parameter.lerp_(source_parameter, tau)


__all__ = ["TargetQCritic"]
