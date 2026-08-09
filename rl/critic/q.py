"""Visual-state action-chunk critics for deterministic visuomotor RL."""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from benchmark.base import MetaObs

from .base import BaseCritic


def _observations(obs):
    if isinstance(obs, MetaObs):
        return (obs,)
    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
        result = tuple(obs)
        if result and all(isinstance(item, MetaObs) for item in result):
            return result
    raise TypeError("Q critic input must be MetaObs or a non-empty MetaObs sequence")


def _activation(name):
    try:
        return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]
    except KeyError as exc:
        raise ValueError("activation must be relu, gelu, or tanh") from exc


class FeatureChunkQHead(nn.Module):
    """Fuse policy features with an ordered, mask-aware action chunk."""

    def __init__(self, *, hidden_dim=256, activation="relu", use_language=False):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not isinstance(use_language, bool):
            raise TypeError("use_language must be bool")
        activation_class = _activation(activation)
        self.use_language = use_language
        self.visual_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_class(),
        )
        self.state_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_class(),
        )
        self.action_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_class(),
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
            nn.LazyLinear(hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, 1),
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
            [
                (action * mask.unsqueeze(-1)).flatten(start_dim=1),
                mask,
            ],
            dim=-1,
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
    """Evaluate action chunks using a policy-local detached feature hook."""

    def __init__(
        self,
        feature_provider,
        *,
        hidden_dim=256,
        activation="relu",
        detach_features=True,
        use_language=False,
    ):
        super().__init__()
        provider = getattr(feature_provider, "critic_features", feature_provider)
        if not callable(provider):
            raise TypeError(
                "feature_provider must be callable or expose critic_features"
            )
        if not isinstance(detach_features, bool):
            raise TypeError("detach_features must be bool")
        self.feature_provider = provider
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
        observations = _observations(obs)
        features = self.feature_provider(observations, context=context)
        if not isinstance(features, Mapping):
            raise TypeError("critic feature hook must return a mapping")
        if "visual" not in features or "state" not in features:
            raise KeyError("Q critic features require visual and state")
        visual = self._feature(features["visual"], name="visual")
        state = self._feature(
            features["state"],
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

        action = torch.as_tensor(
            action,
            dtype=visual.dtype,
            device=visual.device,
        )
        if action.ndim != 3 or action.shape[0] != len(observations):
            raise ValueError(
                "Q critic action must have shape [observation, chunk, action]"
            )
        if action_mask is None:
            action_mask = torch.ones(
                action.shape[:2],
                dtype=torch.bool,
                device=action.device,
            )
        else:
            action_mask = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=action.device,
            )
        return self.head(visual, state, action, action_mask, language)


__all__ = ["FeatureChunkQHead", "PolicyFeatureChunkQCritic"]
