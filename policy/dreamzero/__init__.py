"""
DreamZero policy module for ILStudio.

DreamZero is a world-model-based VLA that jointly predicts future video
frames and robot actions via flow-matching on a Wan2.1 video diffusion
transformer.  Core model code is provided by the ``dreamzero`` git submodule
under ``policy/dreamzero/dreamzero/groot/``.

Interfaces (ILStudio contract):
    load_model(args)              → {'model': DreamZeroPolicy}
    get_data_processor(args, mc)  → DreamZeroProcessor
    get_data_collator(args, mc)   → DreamZeroCollator
    Trainer                       → policy.dreamzero.trainer.Trainer
"""

import os
import torch

from .modeling import DreamZeroPolicy, DreamZeroPolicyConfig
from .data_utils import DreamZeroProcessor, DreamZeroCollator
from .trainer import Trainer


def _looks_like_local_tokenizer_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return any(
        os.path.exists(os.path.join(path, name))
        for name in ("tokenizer_config.json", "spiece.model", "tokenizer.json")
    )


def _resolve_tokenizer_path(tokenizer_path: str, checkpoint_path: str) -> str:
    """Prefer a local tokenizer directory for offline inference."""
    env_override = os.getenv("DREAMZERO_TOKENIZER_PATH", "").strip()
    candidates = []
    if env_override:
        candidates.append(env_override)

    ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    roots = [ckpt_dir, os.path.dirname(ckpt_dir), os.getcwd()]

    if tokenizer_path:
        candidates.append(tokenizer_path)
        for root in roots:
            candidates.append(os.path.join(root, tokenizer_path))

    for root in roots:
        candidates.extend(
            [
                os.path.join(root, "google", "umt5-xxl"),
                os.path.join(root, "tokenizer"),
                os.path.join(root, "tokenizer", "google", "umt5-xxl"),
                os.path.join(root, "pretrained_model", "Wan2.1-I2V-14B-480P", "google", "umt5-xxl"),
            ]
        )

    shared_snapshot_roots = [
        "/inspire/hdd/project/robot-action/public/models/hub/models--google--umt5-xxl/snapshots",
        "/inspire/hdd/global_public/models/hub/models--google--umt5-xxl/snapshots",
        "/inspire/hdd/global_user/wangzheng-240308120196/models/hub/models--google--umt5-xxl/snapshots",
    ]
    for snapshot_root in shared_snapshot_roots:
        if not os.path.isdir(snapshot_root):
            continue
        for snapshot_name in os.listdir(snapshot_root):
            candidates.append(os.path.join(snapshot_root, snapshot_name))

    for candidate in candidates:
        if _looks_like_local_tokenizer_dir(candidate):
            return candidate

    return tokenizer_path


def _patch_post_initialize_no_compile(vla):
    """Wrap ``action_head.post_initialize`` to skip ``torch.compile`` which
    causes ``FailOnRecompileLimitHit`` when input shapes vary at inference."""
    action_head = getattr(vla, "action_head", None)
    if action_head is None:
        return
    _orig = action_head.post_initialize

    def _no_compile_post_init():
        import os
        old = os.environ.get("ENABLE_TENSORRT")
        os.environ["ENABLE_TENSORRT"] = "True"
        try:
            _orig()
        finally:
            if old is None:
                os.environ.pop("ENABLE_TENSORRT", None)
            else:
                os.environ["ENABLE_TENSORRT"] = old

    action_head.post_initialize = _no_compile_post_init


def load_model(args):
    """Load a DreamZero model for training or inference.

    Training:
        Builds a fresh DreamZeroPolicy from YAML ``model_args``.
        If ``dreamzero_pretrained_path`` is set, loads pretrained DreamZero
        weights (safetensors from original training pipeline).

    Inference:
        Loads a saved ILStudio checkpoint via ``from_pretrained``, or a
        native DreamZero checkpoint via ``from_dreamzero_checkpoint``.
    """
    if not args.is_training:
        checkpoint_path = args.model_name_or_path

        has_ilstudio = _is_ilstudio_checkpoint(checkpoint_path)
        if has_ilstudio:
            config = DreamZeroPolicyConfig.from_pretrained(checkpoint_path)
            config.skip_component_loading = True
            model = DreamZeroPolicy.from_pretrained(
                checkpoint_path,
                config=config,
                trust_remote_code=True,
            )
        else:
            model = DreamZeroPolicy.from_dreamzero_checkpoint(
                checkpoint_path, device="cpu",
            )

        dtype = getattr(torch, str(getattr(model.config, "compute_dtype", "bfloat16")), torch.bfloat16)
        model = model.to(device="cuda", dtype=dtype)
        if hasattr(model, "vla") and hasattr(model.vla, "post_initialize"):
            _patch_post_initialize_no_compile(model.vla)
            model.vla.post_initialize()
        model.eval()

        config = model.config
        tokenizer_path = _resolve_tokenizer_path(config.tokenizer_path, checkpoint_path)
        model.data_processor = DreamZeroProcessor(
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            action_horizon=config.action_horizon,
            state_horizon=config.state_horizon,
            num_video_frames=config.num_video_frames,
            image_size=tuple(config.image_size),
            num_views=config.num_views,
        )
        model.data_collator = DreamZeroCollator(
            tokenizer_path=tokenizer_path,
        )
        return {"model": model}

    # --- Training ---
    model_args = getattr(args, "model_args", {})
    pretrained_path = model_args.pop("dreamzero_pretrained_path", "")

    if pretrained_path:
        model = DreamZeroPolicy.from_dreamzero_checkpoint(
            pretrained_path,
            override_config=model_args,
            device="cpu",
        )
    else:
        config = DreamZeroPolicyConfig(**model_args)
        model = DreamZeroPolicy(config=config)

    return {"model": model}


def get_data_processor(args, model_components):
    """Per-sample processor: ILStudio standard → DreamZero format."""
    view_layout = getattr(args, "view_layout", "side_by_side")
    language_prefix = getattr(args, "language_prefix", "")
    embodiment_id = getattr(args, "embodiment_id", 0)
    image_size = getattr(args, "image_size", [256, 256])
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = tuple(image_size)
    else:
        image_size = (256, 256)

    return DreamZeroProcessor(
        max_action_dim=getattr(args, "max_action_dim", 64),
        max_state_dim=getattr(args, "max_state_dim", 64),
        action_horizon=getattr(args, "action_horizon", 24),
        state_horizon=getattr(args, "state_horizon", 16),
        num_video_frames=getattr(args, "num_video_frames", 33),
        image_size=image_size,
        num_views=getattr(args, "num_views", 2),
        view_layout=view_layout,
        embodiment_id=embodiment_id,
        language_prefix=language_prefix,
    )


def get_data_collator(args, model_components):
    """Batch collator with text tokenization."""
    return DreamZeroCollator(
        tokenizer_path=getattr(args, "tokenizer_path", "google/umt5-xxl"),
        max_text_len=getattr(args, "max_text_len", 512),
    )


def _is_ilstudio_checkpoint(path):
    import json as _json
    meta = os.path.join(path, "policy_metadata.json")
    if os.path.exists(meta):
        return True
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg) as f:
                d = _json.load(f)
            return d.get("model_type") == "dreamzero"
        except Exception:
            pass
    return False
