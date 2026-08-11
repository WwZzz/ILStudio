"""Policy-independent state-action Q networks."""

import torch
import torch.nn as nn

from ..base import BaseCritic
from ..common import activation as resolve_activation, observations, states


class StateActionQCritic(BaseCritic):
    """Evaluate flat or chunked actions from robot state."""

    def __init__(self, *, hidden_dim=256, activation="relu"):
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

    def forward(self, obs, action, *, action_mask=None, context=None):
        del context
        items = observations(obs)
        state = states(items, device=next(self.parameters()).device)
        action = torch.as_tensor(action, dtype=state.dtype, device=state.device)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.ndim == 3:
            if action_mask is not None:
                mask = torch.as_tensor(
                    action_mask,
                    dtype=action.dtype,
                    device=action.device,
                )
                if mask.shape != action.shape[:2]:
                    raise ValueError("action_mask must match action chunk prefix")
                action = torch.cat(
                    [(action * mask.unsqueeze(-1)).flatten(start_dim=1), mask],
                    dim=-1,
                )
            else:
                action = action.flatten(start_dim=1)
        if action.ndim != 2 or action.shape[0] != state.shape[0]:
            raise ValueError("critic actions must have shape [batch, action_features]")
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


__all__ = ["StateActionQCritic"]
