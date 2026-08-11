"""Policy-independent state value critic."""

import torch.nn as nn

from ..base import BaseCritic
from ..common import activation as resolve_activation, observations, states


class StateValueCritic(BaseCritic):
    required_observation_fields = frozenset({"state"})

    def __init__(self, *, hidden_dim=256, activation="tanh"):
        super().__init__()
        activation_class = resolve_activation(activation)
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, *, context=None):
        del context
        items = observations(obs)
        return self.net(states(items, device=next(self.parameters()).device)).squeeze(-1)


__all__ = ["StateValueCritic"]
