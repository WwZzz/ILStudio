"""Reusable value-head networks."""

import torch
import torch.nn as nn


class FeatureValueHead(nn.Module):
    def __init__(self, *, hidden_dim=256, use_language=True):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(use_language, bool):
            raise TypeError("use_language must be bool")
        self.visual_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.state_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.language_projection = (
            nn.Sequential(
                nn.LazyLinear(hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            if use_language
            else None
        )
        self.value_head = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, visual, state, language=None):
        if visual.ndim != 2 or state.ndim != 2:
            raise ValueError("critic visual and state features must be rank two")
        if visual.shape[0] != state.shape[0]:
            raise ValueError("critic feature batch sizes must agree")
        features = [self.visual_projection(visual), self.state_projection(state)]
        if language is not None:
            if self.language_projection is None:
                raise ValueError("this critic head was configured without language")
            if language.ndim != 2 or language.shape[0] != visual.shape[0]:
                raise ValueError("critic language features must align with observations")
            features.append(self.language_projection(language))
        return self.value_head(torch.cat(features, dim=-1)).squeeze(-1)


__all__ = ["FeatureValueHead"]
