"""Visual-state critics for VLA and visuomotor policies."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.base import MetaObs

from .base import BaseCritic


def _observations(obs):
    if isinstance(obs, MetaObs):
        return (obs,)
    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
        result = tuple(obs)
        if result and all(isinstance(item, MetaObs) for item in result):
            return result
    raise TypeError("critic input must be MetaObs or a non-empty sequence of MetaObs")


def _states(observations, *, device):
    values = []
    for obs in observations:
        state = obs.state
        if state is None:
            state = obs.state_ee if obs.state_ee is not None else obs.state_joint
        if state is None:
            raise ValueError("critic observation must contain state")
        value = torch.as_tensor(state, dtype=torch.float32, device=device)
        if value.ndim != 1:
            raise ValueError("critic state must be one-dimensional per observation")
        values.append(value)
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("critic state dimensions must agree within a batch")
    return torch.stack(values)


class FeatureValueHead(nn.Module):
    """Fuse visual, state, and optional language features into state values."""

    def __init__(self, *, hidden_dim=256, use_language=True):
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(use_language, bool):
            raise TypeError("use_language must be bool")
        self.use_language = use_language
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
            if use_language else None
        )
        self.value_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, visual, state, language=None):
        if visual.ndim != 2 or state.ndim != 2:
            raise ValueError("critic visual and state features must be rank two")
        if visual.shape[0] != state.shape[0]:
            raise ValueError("critic feature batch sizes must agree")
        features = [
            self.visual_projection(visual),
            self.state_projection(state),
        ]
        if language is not None:
            if self.language_projection is None:
                raise ValueError("this critic head was configured without language")
            if language.ndim != 2 or language.shape[0] != visual.shape[0]:
                raise ValueError("critic language features must align with observations")
            features.append(self.language_projection(language))
        return self.value_head(torch.cat(features, dim=-1)).squeeze(-1)


class PolicyFeatureCritic(BaseCritic):
    """Build values from features exposed by a policy-local RL adapter."""

    def __init__(
        self,
        feature_provider,
        *,
        hidden_dim=256,
        detach_features=True,
        use_language=True,
    ):
        super().__init__()
        provider = getattr(feature_provider, "critic_features", feature_provider)
        if not callable(provider):
            raise TypeError("feature_provider must be callable or expose critic_features")
        if not isinstance(detach_features, bool):
            raise TypeError("detach_features must be bool")
        self.feature_provider = provider
        self.detach_features = detach_features
        self.head = FeatureValueHead(
            hidden_dim=hidden_dim,
            use_language=use_language,
        )

    def forward(self, obs, *, context=None):
        observations = _observations(obs)
        features = self.feature_provider(observations, context=context)
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
            state = _states(observations, device=visual.device)
        elif not torch.is_tensor(state):
            state = torch.as_tensor(
                state, dtype=visual.dtype, device=visual.device
            )
        return self.head(visual, state, language)


class DinoStateCritic(BaseCritic):
    """Independent frozen-DINO visual critic with a trainable state/value head."""

    def __init__(
        self,
        *,
        visual_encoder=None,
        model_name="vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        freeze_visual=True,
        image_size=224,
        hidden_dim=256,
    ):
        super().__init__()
        if visual_encoder is None:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "DinoStateCritic requires timm or an explicit visual_encoder"
                ) from exc
            visual_encoder = timm.create_model(
                model_name,
                pretrained=bool(pretrained),
                num_classes=0,
                img_size=image_size,
            )
        if not isinstance(visual_encoder, nn.Module):
            raise TypeError("visual_encoder must be a torch module")
        if not isinstance(freeze_visual, bool):
            raise TypeError("freeze_visual must be bool")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.visual_encoder = visual_encoder
        self.freeze_visual = freeze_visual
        self.image_size = image_size
        self.head = FeatureValueHead(hidden_dim=hidden_dim, use_language=False)
        if freeze_visual:
            self.visual_encoder.requires_grad_(False)
            self.visual_encoder.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        return self

    def _images(self, observations):
        images = []
        camera_counts = []
        for obs in observations:
            if obs.image is None:
                raise ValueError("DinoStateCritic requires observation images")
            value = torch.as_tensor(obs.image, device=self.image_mean.device)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError(
                    "critic images must have shape [camera, 3, height, width]"
                )
            value = value.float()
            if value.numel() and float(value.detach().max()) > 1.0:
                value = value / 255.0
            images.append(value)
            camera_counts.append(value.shape[0])
        images = torch.cat(images)
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (images - self.image_mean) / self.image_std, camera_counts

    @staticmethod
    def _encoder_output(value):
        if isinstance(value, Mapping):
            for key in ("x_norm_clstoken", "pooler_output", "features"):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value):
            raise TypeError("visual encoder must return tensor features")
        while value.ndim > 2:
            value = value.mean(dim=1)
        if value.ndim != 2:
            raise ValueError("visual encoder output must contain one feature vector")
        return value

    def forward(self, obs, *, context=None):
        del context
        observations = _observations(obs)
        images, camera_counts = self._images(observations)
        if self.freeze_visual:
            with torch.no_grad():
                encoded = self.visual_encoder(images)
            encoded = encoded.detach()
        else:
            encoded = self.visual_encoder(images)
        encoded = self._encoder_output(encoded)
        visual = []
        start = 0
        for count in camera_counts:
            visual.append(encoded[start : start + count].mean(dim=0))
            start += count
        visual = torch.stack(visual)
        state = _states(observations, device=visual.device)
        return self.head(visual, state)
