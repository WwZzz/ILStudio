"""Q critics built from named policy features and action chunks."""

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..base import BaseCritic
from ..common import activation as resolve_activation, observations, states
from ..feature_hook import PolicyFeatureExtractor


class FeatureChunkQHead(nn.Module):
    def __init__(self, *, hidden_dim=256, activation="relu", use_language=False):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not isinstance(use_language, bool):
            raise TypeError("use_language must be bool")
        activation_class = resolve_activation(activation)
        self.visual_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), activation_class()
        )
        self.state_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), activation_class()
        )
        self.action_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), activation_class()
        )
        self.language_projection = (
            nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation_class(),
            )
            if use_language
            else None
        )
        self.q_head = nn.Sequential(
            nn.LazyLinear(hidden_dim), activation_class(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, visual, state, action, action_mask, language=None):
        if visual.ndim != 2 or state.ndim != 2:
            raise ValueError("visual and state features must be rank two")
        if action.ndim != 3:
            raise ValueError("action chunks must have shape [batch, chunk, action]")
        if action_mask.ndim != 2 or action_mask.shape != action.shape[:2]:
            raise ValueError("action mask must have shape [batch, chunk]")
        batch_size = visual.shape[0]
        if state.shape[0] != batch_size or action.shape[0] != batch_size:
            raise ValueError("Q critic batch dimensions must agree")
        mask = action_mask.to(dtype=action.dtype)
        action_input = torch.cat(
            [(action * mask.unsqueeze(-1)).flatten(start_dim=1), mask], dim=-1
        )
        features = [
            self.visual_projection(visual),
            self.state_projection(state),
            self.action_projection(action_input),
        ]
        if language is not None:
            if self.language_projection is None:
                raise ValueError("this Q head was configured without language")
            if language.ndim != 2 or language.shape[0] != batch_size:
                raise ValueError("language features must align with Q inputs")
            features.append(self.language_projection(language))
        return self.q_head(torch.cat(features, dim=-1)).squeeze(-1)


class PolicyFeatureChunkQCritic(BaseCritic):
    def __init__(
        self,
        policy_adapter,
        *,
        feature_hooks=None,
        hidden_dim=256,
        activation="relu",
        detach_features=True,
        use_language=False,
    ):
        super().__init__()
        feature_method = (
            PolicyFeatureExtractor(policy_adapter, feature_hooks)
            if feature_hooks is not None
            else getattr(policy_adapter, "features", None)
        )
        if not callable(feature_method):
            raise TypeError("policy_adapter must expose features()")
        if not isinstance(detach_features, bool):
            raise TypeError("detach_features must be bool")
        self.feature_extractor = feature_method
        self.detach_features = detach_features
        self.head = FeatureChunkQHead(
            hidden_dim=hidden_dim,
            activation=activation,
            use_language=use_language,
        )

    @staticmethod
    def _feature(value, *, name, device=None, dtype=torch.float32):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=dtype, device=device)
        elif device is not None:
            value = value.to(device=device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(f"{name} features must be rank two")
        return value

    def forward(self, obs, action, *, action_mask=None, context=None):
        items = observations(obs)
        features = self.feature_extractor(items, context=context)
        if not isinstance(features, Mapping):
            raise TypeError("critic feature hook must return a mapping")
        if "visual" not in features:
            raise KeyError("Q critic features require visual")
        visual = self._feature(features["visual"], name="visual")
        state = self._feature(
            features.get("state", states(items)),
            name="state",
            device=visual.device,
            dtype=visual.dtype,
        )
        language = features.get("language")
        if language is not None:
            language = self._feature(
                language,
                name="language",
                device=visual.device,
                dtype=visual.dtype,
            )
        if self.detach_features:
            visual = visual.detach()
            state = state.detach()
            if language is not None:
                language = language.detach()
        action = torch.as_tensor(action, dtype=visual.dtype, device=visual.device)
        if action.ndim != 3 or action.shape[0] != len(items):
            raise ValueError(
                "Q critic action must have shape [observation, chunk, action]"
            )
        if action_mask is None:
            action_mask = torch.ones(
                action.shape[:2], dtype=torch.bool, device=action.device
            )
        else:
            action_mask = torch.as_tensor(
                action_mask, dtype=torch.bool, device=action.device
            )
        return self.head(visual, state, action, action_mask, language)


__all__ = ["FeatureChunkQHead", "PolicyFeatureChunkQCritic"]
