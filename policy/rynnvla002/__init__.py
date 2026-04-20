"""
RynnVLA-002 policy module for ILStudio.

RynnVLA-002 is an autoregressive action world model that unifies VLA and
world model on a Chameleon 7B backbone.  It uses VQGAN-discretized image
tokens, 256-bin discretized action/state tokens in the sequence, and an
auxiliary continuous action head (TransformerEncoder + MLP) for regression.

Interfaces (ILStudio contract):
    load_model(args)              → {'model': RynnVLA002Policy, 'config': ...}
    get_data_processor(args, mc)  → RynnVLA002Processor
    get_data_collator(args, mc)   → RynnVLA002Collator
    Trainer                       → policy.rynnvla002.trainer.Trainer
"""

import os
import json
import torch
from pathlib import Path
from loguru import logger

from .modeling import RynnVLA002Policy, RynnVLA002PolicyConfig
from .data_utils import RynnVLA002Processor, RynnVLA002Collator
from .trainer import Trainer
from .hf_assets import maybe_download_rynnvla_ckpts

__all__ = [
    "RynnVLA002Policy",
    "RynnVLA002PolicyConfig",
    "Trainer",
    "load_model",
    "get_data_processor",
    "get_data_collator",
]

_POLICY_DIR = Path(__file__).resolve().parent


def _expand_paths_relative_to_policy(config) -> None:
    """Resolve README-style paths relative to ``policy/rynnvla002/`` or cwd.

    Hugging Face ``org/name`` ids are left unchanged when they do not exist as
    local paths under those bases.
    """
    for attr in ("pretrained_path", "tokenizer_path", "chameleon_tokenizer_dir"):
        raw = getattr(config, attr, None)
        if raw is None:
            continue
        p = str(raw).strip()
        if not p or os.path.isabs(p):
            continue
        for base in (_POLICY_DIR, Path.cwd()):
            cand = base / p
            if cand.is_dir() or cand.is_file():
                setattr(config, attr, str(cand.resolve()))
                break


def _resolve_chameleon_tokenizer_dir(config) -> str:
    """Locate the Chameleon VQGAN tokenizer directory.

    Search order:
    1. Explicit ``chameleon_tokenizer_dir`` in config
    2. ``<pretrained_path>/../chameleon/tokenizer``
    3. Well-known shared locations on the cluster
    """
    if config.chameleon_tokenizer_dir and os.path.isdir(config.chameleon_tokenizer_dir):
        return config.chameleon_tokenizer_dir

    if config.pretrained_path:
        candidate = os.path.join(
            os.path.dirname(config.pretrained_path), "chameleon", "tokenizer"
        )
        if os.path.isdir(candidate):
            return candidate

    repo_root = Path(__file__).resolve().parent / "RynnVLA-002" / "rynnvla-002" / "ckpts"
    candidate = str(repo_root / "chameleon" / "tokenizer")
    if os.path.isdir(candidate):
        return candidate

    return config.chameleon_tokenizer_dir or ""


def _resolve_tokenizer_path(config) -> str:
    """Locate the Lumina-mGPT HF tokenizer directory."""
    if config.tokenizer_path and os.path.isdir(config.tokenizer_path):
        return config.tokenizer_path

    if config.pretrained_path:
        for subdir in [
            "tokenizer",
            os.path.join(os.path.dirname(config.pretrained_path), "tokenizer"),
        ]:
            if os.path.isdir(subdir):
                return subdir

    repo_root = Path(__file__).resolve().parent / "RynnVLA-002" / "rynnvla-002" / "ckpts"
    lumina = repo_root / "models--Alpha-VLLM--Lumina-mGPT-7B-768"
    for snap_root in [lumina / "snapshots"]:
        if snap_root.is_dir():
            for snap in snap_root.iterdir():
                if (snap / "tokenizer.json").is_file():
                    return str(snap)

    return config.tokenizer_path or ""


def _is_ilstudio_checkpoint(path: str) -> bool:
    meta = os.path.join(path, "policy_metadata.json")
    if os.path.exists(meta):
        return True
    cfg_file = os.path.join(path, "config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file) as f:
                d = json.load(f)
            return d.get("model_type") == "rynnvla002"
        except Exception:
            pass
    return False


def load_model(args) -> dict:
    """Load a RynnVLA-002 model for training or inference.

    Training:
        Builds a fresh RynnVLA002Policy from YAML ``model_args`` and
        loads pretrained Chameleon weights from ``pretrained_path``.

    Inference:
        Loads a saved ILStudio checkpoint or an upstream RynnVLA-002 checkpoint.
    """
    is_training = getattr(args, "is_training", False)

    if not is_training:
        checkpoint_path = getattr(args, "model_name_or_path", "")

        if _is_ilstudio_checkpoint(checkpoint_path):
            config = RynnVLA002PolicyConfig.from_pretrained(checkpoint_path)
            maybe_download_rynnvla_ckpts(policy_dir=_POLICY_DIR, config=config)
            model = RynnVLA002Policy(config)
            state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                sd = torch.load(state_dict_path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
        else:
            model_args = dict(getattr(args, "model_args", {}) or {})
            model_args.setdefault("pretrained_path", checkpoint_path)
            config = RynnVLA002PolicyConfig(**model_args)
            maybe_download_rynnvla_ckpts(policy_dir=_POLICY_DIR, config=config)
            _expand_paths_relative_to_policy(config)
            config.chameleon_tokenizer_dir = _resolve_chameleon_tokenizer_dir(config)
            config.tokenizer_path = _resolve_tokenizer_path(config)
            model = RynnVLA002Policy(config)
            model.load_pretrained_chameleon(checkpoint_path)

        model = model.to(dtype=torch.bfloat16, device="cuda")
        model.eval()

        config.chameleon_tokenizer_dir = _resolve_chameleon_tokenizer_dir(config)
        config.tokenizer_path = _resolve_tokenizer_path(config)

        image_size = getattr(args, "image_size", None) or config.image_size or [256, 256]
        model.data_processor = RynnVLA002Processor(
            num_views=2 if config.with_wrist else 1,
            image_size=tuple(image_size),
            history_len=config.history_len,
            with_wrist=config.with_wrist,
            with_state=config.with_state,
            time_horizon=config.time_horizon,
        )
        model.data_collator = RynnVLA002Collator()

        return {"model": model, "config": config}

    # --- Training ---
    model_args = dict(getattr(args, "model_args", {}) or {})
    config = RynnVLA002PolicyConfig(**model_args)
    maybe_download_rynnvla_ckpts(policy_dir=_POLICY_DIR, config=config)
    _expand_paths_relative_to_policy(config)
    config.chameleon_tokenizer_dir = _resolve_chameleon_tokenizer_dir(config)
    config.tokenizer_path = _resolve_tokenizer_path(config)

    model = RynnVLA002Policy(config)
    pretrained = (getattr(config, "pretrained_path", None) or "").strip()
    if not pretrained:
        raise ValueError(
            "RynnVLA-002 training requires a non-empty `pretrained_path` in "
            "`configs/policy/rynnvla002.yaml` (or merged `model_args`). "
            "Point it to a local Chameleon / RynnVLA-002 checkpoint directory, "
            "or a Hugging Face repo id. See `policy/rynnvla002/README.md`."
        )
    model.load_pretrained_chameleon(pretrained)

    return {"model": model, "config": config}


def get_data_processor(args, model_components):
    """Return the per-sample data processor."""
    config = model_components.get("config")
    if config is None:
        config = RynnVLA002PolicyConfig(**dict(getattr(args, "model_args", {}) or {}))

    image_size = getattr(args, "image_size", None) or config.image_size or [256, 256]
    return RynnVLA002Processor(
        num_views=2 if config.with_wrist else 1,
        image_size=tuple(image_size),
        history_len=config.history_len,
        with_wrist=config.with_wrist,
        with_state=config.with_state,
        time_horizon=config.time_horizon,
    )


def get_data_collator(args, model_components):
    """Return the batch collator."""
    return RynnVLA002Collator()
