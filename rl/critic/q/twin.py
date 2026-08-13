"""Twin-Q ownership and target-network maintenance."""

import copy
import importlib

import torch
from torch.nn.parameter import UninitializedParameter

from ..base import BaseCritic


class TwinQCritic(BaseCritic):
    def __init__(self, *, q_type="rl.critic.StateActionQCritic", q_args=None):
        super().__init__()
        if not isinstance(q_type, str) or "." not in q_type:
            raise TypeError("q_type must be a fully-qualified import path")
        module_name, symbol_name = q_type.rsplit(".", 1)
        q_class = getattr(importlib.import_module(module_name), symbol_name)
        self.q1 = q_class(**dict(q_args or {}))
        self.q2 = q_class(**dict(q_args or {}))
        if not isinstance(self.q1, BaseCritic) or not isinstance(self.q2, BaseCritic):
            raise TypeError("TwinQCritic q_type must construct BaseCritic")
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        self.target_q1.eval()
        self.target_q2.eval()

    def parameters(self, recurse=True):
        yield from self.q1.parameters(recurse=recurse)
        yield from self.q2.parameters(recurse=recurse)

    def train(self, mode=True):
        super().train(mode)
        self.target_q1.eval()
        self.target_q2.eval()
        return self

    def forward(self, obs, action, *, action_mask=None, context=None):
        kwargs = {"action_mask": action_mask, "context": context}
        return self.q1(obs, action, **kwargs), self.q2(obs, action, **kwargs)

    def target(self, obs, action, *, action_mask=None, context=None):
        if any(
            isinstance(parameter, UninitializedParameter)
            for parameter in self.target_q1.parameters()
        ):
            self.target_q1.load_state_dict(self.q1.state_dict())
            self.target_q2.load_state_dict(self.q2.state_dict())
            self.target_q1.requires_grad_(False)
            self.target_q2.requires_grad_(False)
        kwargs = {"action_mask": action_mask, "context": context}
        with torch.no_grad():
            return (
                self.target_q1(obs, action, **kwargs),
                self.target_q2(obs, action, **kwargs),
            )

    def parameter_groups(self):
        return {
            "critic1": tuple(self.q1.parameters()),
            "critic2": tuple(self.q2.parameters()),
        }

    def soft_update(self, tau):
        tau = float(tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        with torch.no_grad():
            for target, source in (
                (self.target_q1, self.q1),
                (self.target_q2, self.q2),
            ):
                for target_parameter, source_parameter in zip(
                    target.parameters(), source.parameters()
                ):
                    target_parameter.lerp_(source_parameter, tau)


__all__ = ["TwinQCritic"]
