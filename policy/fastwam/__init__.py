"""ILStudio policy module for FastWAM (Fast World-Action Model).

Supports three attention modes via ``attention_mode`` in :class:`~policy.fastwam.modeling.FastWAMPolicyConfig`:
- ``original``: action attends only to first-frame video tokens (fastwam.py)
- ``joint``: action attends to all video tokens (fastwam_joint.py)
- ``idm``: inverse dynamics model with teacher-forcing conditioning (fastwam_idm.py)

WAN backbone weights are downloaded automatically on first use via ``huggingface_hub`` or ``modelscope``
(controlled by ``download_source`` on the config).
"""

from pathlib import Path

import torch
from loguru import logger

from .data_utils import FastWAMDataCollator, FastWAMDataProcessor
from .modeling import FastWAMPolicy, FastWAMPolicyConfig
from .trainer import Trainer

__all__ = [
    "FastWAMPolicy",
    "FastWAMPolicyConfig",
    "Trainer",
    "get_data_collator",
    "get_data_processor",
    "load_model",
]


def _resolve_checkpoint_dir(checkpoint_path: str | None) -> Path | None:
    """Return the directory containing ``config.json`` next to a checkpoint, or None."""
    if not checkpoint_path:
        return None
    p = Path(checkpoint_path).expanduser()
    if p.is_file():
        p = p.parent
    if p.name.startswith("checkpoint-"):
        p = p.parent
    if (p / "config.json").is_file():
        return p
    return None


def load_model(args) -> dict:
    """ILStudio entry: build :class:`FastWAMPolicy` from ``args.model_args`` + optional checkpoint."""
    is_training = getattr(args, "is_training", False)
    checkpoint_path = getattr(args, "model_name_or_path", None)

    # Inference from saved checkpoint: load the full config via from_pretrained
    # so nested dicts (video_dit_config, etc.) are recovered without YAML.
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path) if not is_training else None
    if ckpt_dir is not None:
        config = FastWAMPolicyConfig.from_pretrained(str(ckpt_dir))
        config.inference_mode = True
    else:
        cfg = dict(getattr(args, "model_args", None) or {})
        cfg["inference_mode"] = not is_training
        config = FastWAMPolicyConfig.from_merged_args(cfg)

    policy = FastWAMPolicy(config)

    if config.inference_mode and checkpoint_path is not None:
        wp = Path(checkpoint_path).expanduser()
        weight_file = str(wp) if wp.suffix == ".pt" else str(wp / "fastwam_checkpoint.pt")
        logger.info(f"Loading FastWAM checkpoint weights: {weight_file}")
        policy.load_checkpoint(weight_file)

        policy.eval()
        policy.to("cuda" if torch.cuda.is_available() else "cpu")

        image_size = getattr(args, "image_size", None) or config.image_size or [224, 224]
        policy.data_processor = FastWAMDataProcessor(
            num_views=config.num_views,
            image_size=image_size,
        )
        policy.data_collator = FastWAMDataCollator(is_training=False)

    return {"model": policy, "config": config}


def get_data_processor(args, model_components):
    """Return the per-sample data processor."""
    cfg = model_components.get("config")
    if cfg is None:
        cfg = FastWAMPolicyConfig.from_merged_args(dict(getattr(args, "model_args", None) or {}))
    num_views = cfg.num_views
    image_size = getattr(args, "image_size", None) or cfg.image_size or [224, 224]
    return FastWAMDataProcessor(
        num_views=num_views,
        image_size=image_size,
    )


def get_data_collator(args, model_components):
    """Return the batch collator."""
    is_training = getattr(args, "is_training", True)
    return FastWAMDataCollator(is_training=is_training)
