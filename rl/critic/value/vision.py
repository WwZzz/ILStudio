"""Independent visual value critics."""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseCritic
from ..common import observations, states
from .heads import FeatureValueHead


class DinoStateCritic(BaseCritic):
    """Frozen-DINO visual encoder plus a trainable state/value head."""

    required_observation_fields = frozenset({"image", "state"})

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
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.visual_encoder.eval()
        return self

    def _images(self, items):
        images = []
        camera_counts = []
        for obs in items:
            if obs.image is None:
                raise ValueError("DinoStateCritic requires observation images")
            value = torch.as_tensor(obs.image, device=self.image_mean.device)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.shape[1] != 3:
                raise ValueError("critic images must have shape [camera, 3, height, width]")
            value = value.float()
            if value.numel() and float(value.detach().max()) > 1.0:
                value = value / 255.0
            images.append(value)
            camera_counts.append(value.shape[0])
        images = F.interpolate(
            torch.cat(images),
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

    def _visual_features(self, items):
        images, camera_counts = self._images(items)
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
        return torch.stack(visual)

    def forward(self, obs, *, context=None):
        del context
        items = observations(obs)
        visual = self._visual_features(items)
        return self.head(visual, states(items, device=visual.device))


class DinoVisualCritic(DinoStateCritic):
    """Independent DINO critic that intentionally ignores robot state."""

    required_observation_fields = frozenset({"image"})

    def __init__(self, *, hidden_dim=256, **kwargs):
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        self.head = nn.Sequential(
            nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, *, context=None):
        del context
        return self.head(self._visual_features(observations(obs))).squeeze(-1)


__all__ = ["DinoStateCritic", "DinoVisualCritic"]
