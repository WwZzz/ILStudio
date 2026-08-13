"""Composable Q-and-value critic ownership for offline RL algorithms."""

import importlib

from ..base import BaseCritic


def _critic(type_path, args, *, name):
    if not isinstance(type_path, str) or "." not in type_path:
        raise TypeError(f"{name}_type must be a fully-qualified import path")
    module_name, symbol_name = type_path.rsplit(".", 1)
    critic_class = getattr(importlib.import_module(module_name), symbol_name)
    result = critic_class(**dict(args or {}))
    if not isinstance(result, BaseCritic):
        raise TypeError(f"{name}_type must construct BaseCritic")
    return result


class QVCompositeCritic(BaseCritic):
    """Own a target-enabled Q critic and a separate state-value critic."""

    def __init__(
        self,
        *,
        q_type="rl.critic.TwinQCritic",
        q_args=None,
        value_type="rl.critic.StateValueCritic",
        value_args=None,
    ):
        super().__init__()
        self.q_critic = _critic(q_type, q_args, name="q")
        self.value_critic = _critic(value_type, value_args, name="value")
        for name in ("target", "parameter_groups", "soft_update"):
            if not callable(getattr(self.q_critic, name, None)):
                raise TypeError(f"QVCompositeCritic Q module must provide {name}()")

    def parameters(self, recurse=True):
        yield from self.q_critic.parameters(recurse=recurse)
        yield from self.value_critic.parameters(recurse=recurse)

    def forward(self, obs, action, *, action_mask=None, context=None):
        return self.q_critic(
            obs,
            action,
            action_mask=action_mask,
            context=context,
        )

    def target(self, obs, action, *, action_mask=None, context=None):
        return self.q_critic.target(
            obs,
            action,
            action_mask=action_mask,
            context=context,
        )

    def value(self, obs, *, context=None):
        return self.value_critic(obs, context=context)

    def parameter_groups(self):
        groups = dict(self.q_critic.parameter_groups())
        if "value" in groups:
            raise ValueError("Q critic parameter groups cannot be named value")
        groups["value"] = tuple(self.value_critic.parameters())
        return groups

    def soft_update(self, tau):
        self.q_critic.soft_update(tau)


__all__ = ["QVCompositeCritic"]
