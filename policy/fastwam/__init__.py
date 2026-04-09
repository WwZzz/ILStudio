"""ILStudio policy module for FastWAM (Fast World-Action Model).

Supports three attention modes via ``attention_mode`` config key:
- ``original``: action attends only to first-frame video tokens (fastwam.py)
- ``joint``: action attends to all video tokens (fastwam_joint.py)
- ``idm``: inverse dynamics model with teacher-forcing conditioning (fastwam_idm.py)

Other notable config keys:
- ``generate_video``: if True, inference produces both video and action
- ``download_source``: ``"huggingface"`` or ``"modelscope"`` for WAN weight download

WAN backbone weights are downloaded automatically on first use via
``huggingface_hub`` or ``modelscope`` (controlled by ``download_source``).
"""

import os
import sys
from pathlib import Path

import torch
from loguru import logger

from .modeling import FastWAMPolicy, build_fastwam_model
from .data_utils import FastWAMDataProcessor, FastWAMDataCollator
from .trainer import Trainer


def load_model(args) -> dict:
    """ILStudio entry point: build or load a FastWAM policy.

    Returns:
        dict with key ``model`` -> :class:`FastWAMPolicy`.
    """
    model_args = getattr(args, "model_args", {}) or {}

    if args.is_training:
        return _load_for_training(args, model_args)
    return _load_for_inference(args, model_args)


def _inject_action_dim(dit_config: dict, args) -> dict:
    if "action_dim" not in dit_config:
        dit_config["action_dim"] = getattr(args, "action_dim", 7)
    return dit_config


def _resolve_action_horizon(model_args: dict, args) -> int:
    """Resolve action_horizon: explicit value > task chunk_size > default 16."""
    ah = model_args.get("action_horizon")
    if ah is not None:
        return int(ah)
    return int(getattr(args, "chunk_size", 16))


def _load_for_training(args, model_args: dict) -> dict:
    attention_mode = model_args.get("attention_mode", "original")
    generate_video = model_args.get("generate_video", False)

    video_dit_config = _inject_action_dim(model_args.get("video_dit_config", {}), args)
    action_dit_config = _inject_action_dim(model_args.get("action_dit_config", {}), args)

    policy = build_fastwam_model(
        attention_mode=attention_mode,
        model_id=model_args.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=model_args.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        tokenizer_max_len=model_args.get("tokenizer_max_len", 128),
        load_text_encoder=model_args.get("load_text_encoder", True),
        proprio_dim=model_args.get("proprio_dim", None),
        redirect_common_files=model_args.get("redirect_common_files", True),
        mot_checkpoint_mixed_attn=model_args.get("mot_checkpoint_mixed_attn", True),
        action_dit_pretrained_path=model_args.get("action_dit_pretrained_path", None),
        skip_dit_load_from_pretrain=model_args.get("skip_dit_load_from_pretrain", False),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        video_scheduler=model_args.get("video_scheduler", {}),
        action_scheduler=model_args.get("action_scheduler", {
            "train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000,
        }),
        loss=model_args.get("loss", {}),
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        generate_video=generate_video,
        action_horizon=_resolve_action_horizon(model_args, args),
        num_video_frames=model_args.get("num_video_frames", 17),
        num_inference_steps=model_args.get("num_inference_steps", 20),
        sigma_shift=model_args.get("sigma_shift", None),
        seed=model_args.get("seed", None),
        download_source=model_args.get("download_source", None),
    )

    return {"model": policy}


def _load_for_inference(args, model_args: dict) -> dict:
    checkpoint_path = getattr(args, "model_name_or_path", None)
    attention_mode = model_args.get("attention_mode", "original")
    generate_video = model_args.get("generate_video", False)

    video_dit_config = _inject_action_dim(model_args.get("video_dit_config", {}), args)
    action_dit_config = _inject_action_dim(model_args.get("action_dit_config", {}), args)

    policy = build_fastwam_model(
        attention_mode=attention_mode,
        model_id=model_args.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=model_args.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        tokenizer_max_len=model_args.get("tokenizer_max_len", 128),
        load_text_encoder=model_args.get("load_text_encoder", True),
        proprio_dim=model_args.get("proprio_dim", None),
        redirect_common_files=model_args.get("redirect_common_files", True),
        mot_checkpoint_mixed_attn=model_args.get("mot_checkpoint_mixed_attn", True),
        action_dit_pretrained_path=model_args.get("action_dit_pretrained_path", None),
        skip_dit_load_from_pretrain=True,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        video_scheduler=model_args.get("video_scheduler", {}),
        action_scheduler=model_args.get("action_scheduler", {
            "train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000,
        }),
        loss=model_args.get("loss", {}),
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        generate_video=generate_video,
        action_horizon=_resolve_action_horizon(model_args, args),
        num_video_frames=model_args.get("num_video_frames", 17),
        num_inference_steps=model_args.get("num_inference_steps", 20),
        sigma_shift=model_args.get("sigma_shift", None),
        seed=model_args.get("seed", None),
        download_source=model_args.get("download_source", None),
    )

    if checkpoint_path is not None:
        logger.info(f"Loading FastWAM checkpoint: {checkpoint_path}")
        policy.load_checkpoint(checkpoint_path)

    policy.eval()
    policy.to("cuda" if torch.cuda.is_available() else "cpu")

    image_size = getattr(args, "image_size", [224, 224])
    if isinstance(image_size, (list, tuple)):
        h, w = image_size[0], image_size[1]
    else:
        h = w = int(image_size)
    policy.data_processor = FastWAMDataProcessor(
        num_views=model_args.get("num_views", 1),
        horizon=model_args.get("horizon", 1),
        image_height=h,
        image_width=w,
    )
    policy.data_collator = FastWAMDataCollator(is_training=False)

    return {"model": policy}


def get_data_processor(args, model_components):
    """Return the per-sample data processor."""
    model_args = getattr(args, "model_args", {}) or {}
    num_views = model_args.get("num_views", 1)
    horizon = model_args.get("horizon", 1)
    image_size = getattr(args, "image_size", [224, 224])
    if isinstance(image_size, (list, tuple)):
        h, w = image_size[0], image_size[1]
    else:
        h = w = int(image_size)
    return FastWAMDataProcessor(
        num_views=num_views,
        horizon=horizon,
        image_height=h,
        image_width=w,
    )


def get_data_collator(args, model_components):
    """Return the batch collator."""
    is_training = getattr(args, "is_training", True)
    return FastWAMDataCollator(is_training=is_training)
