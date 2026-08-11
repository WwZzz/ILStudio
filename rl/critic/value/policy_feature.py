"""Value critics built from named policy features."""

from collections.abc import Mapping

import torch

from ..base import BaseCritic
from ..common import observations, states
from ..feature_hook import PolicyFeatureExtractor
from .heads import FeatureValueHead


class PolicyFeatureCritic(BaseCritic):
    def __init__(
        self,
        policy_adapter,
        *,
        feature_hooks=None,
        hidden_dim=256,
        detach_features=True,
        use_language=True,
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
        self.head = FeatureValueHead(
            hidden_dim=hidden_dim,
            use_language=use_language,
        )

    def forward(self, obs, *, context=None):
        items = observations(obs)
        features = self.feature_extractor(items, context=context)
        if not isinstance(features, Mapping) or "visual" not in features:
            raise TypeError("critic feature hook must return a mapping with visual")
        visual = features["visual"]
        if not torch.is_tensor(visual):
            visual = torch.as_tensor(visual, dtype=torch.float32)
        if visual.ndim == 1:
            visual = visual.unsqueeze(0)
        language = features.get("language")
        if language is not None and not torch.is_tensor(language):
            language = torch.as_tensor(
                language, dtype=visual.dtype, device=visual.device
            )
        if language is not None and language.ndim == 1:
            language = language.unsqueeze(0)
        if self.detach_features:
            visual = visual.detach()
            if language is not None:
                language = language.detach()
        state = features.get("state")
        if state is None:
            state = states(items, device=visual.device)
        elif not torch.is_tensor(state):
            state = torch.as_tensor(
                state, dtype=visual.dtype, device=visual.device
            )
        return self.head(visual, state, language)


__all__ = ["PolicyFeatureCritic"]
