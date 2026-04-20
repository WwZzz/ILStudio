from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class FastWAMPolicyConfig(PretrainedConfig):
    """Serializable policy configuration; mirrors ``configs/policy/fastwam.yaml`` ``args``."""

    model_type = "fastwam_policy"

    def __init__(
        self,
        attention_mode: str = "original",
        action_dim: int = 7,
        state_dim: Optional[int] = None,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 128,
        load_text_encoder: bool = True,
        redirect_common_files: bool = True,
        download_source: Optional[str] = None,
        proprio_dim: Optional[int] = None,
        mot_checkpoint_mixed_attn: bool = True,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        inference_mode: bool = False,
        video_dit_config: Optional[Dict[str, Any]] = None,
        action_dit_config: Optional[Dict[str, Any]] = None,
        video_scheduler: Optional[Dict[str, Any]] = None,
        action_scheduler: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        generate_video: bool = False,
        chunk_size: int = 16,
        num_video_frames: int = 17,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        num_views: int = 1,
        horizon: Optional[int] = None,
        image_size: Optional[Any] = None,
        **kwargs,
    ):
        self.attention_mode = attention_mode
        self.action_dim = int(action_dim)
        self.state_dim = state_dim
        self.model_id = model_id
        self.tokenizer_model_id = tokenizer_model_id
        self.tokenizer_max_len = tokenizer_max_len
        self.load_text_encoder = load_text_encoder
        self.redirect_common_files = redirect_common_files
        self.download_source = download_source
        self.proprio_dim = proprio_dim
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.action_dit_pretrained_path = action_dit_pretrained_path
        self.skip_dit_load_from_pretrain = skip_dit_load_from_pretrain
        self.inference_mode = inference_mode
        self.video_dit_config = deepcopy(video_dit_config) if video_dit_config else {}
        self.action_dit_config = deepcopy(action_dit_config) if action_dit_config else {}
        self.video_scheduler = deepcopy(video_scheduler) if video_scheduler else {}
        self.action_scheduler = deepcopy(action_scheduler) if action_scheduler else {}
        self.loss = deepcopy(loss) if loss else {}
        self.generate_video = generate_video
        self.chunk_size = int(chunk_size)
        self.num_video_frames = num_video_frames
        self.num_inference_steps = num_inference_steps
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.num_views = num_views
        self.horizon = horizon
        self.image_size = image_size
        super().__init__(**kwargs)

    def dit_configs_with_action_dim(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return video/action DiT config dicts with ``action_dim`` filled from this config."""
        vd = deepcopy(self.video_dit_config)
        ad = deepcopy(self.action_dit_config)
        adim = self.action_dim
        if adim is not None:
            vd.setdefault("action_dim", adim)
            ad.setdefault("action_dim", adim)
        return vd, ad

    @classmethod
    def from_merged_args(cls, merged: Dict[str, Any]) -> FastWAMPolicyConfig:
        """Build config from ILStudio-merged ``args.model_args`` + task overrides (flat dict)."""
        sig = inspect.signature(cls.__init__)
        valid = {k for k in sig.parameters if k not in ("self", "kwargs")}
        int_keys = {
            "action_dim",
            "state_dim",
            "chunk_size",
            "tokenizer_max_len",
            "num_video_frames",
            "num_inference_steps",
            "num_views",
            "horizon",
        }
        kwargs: Dict[str, Any] = {}
        for k, v in merged.items():
            if k not in valid:
                continue
            if v is None:
                continue
            if k in int_keys:
                kwargs[k] = int(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)


def _setup_fastwam_import_path() -> None:
    """Ensure the upstream ``fastwam`` package is importable.

    The upstream repo is a **git submodule** next to this file (same layout as
    ``policy/openpi/openpi``). Prefer a local editable install::

        git submodule update --init --recursive
        pip install -e policy/fastwam/FastWAM

    If the package is not installed, ``src/`` from that submodule is prepended
    to ``sys.path`` (development convenience only).
    """
    try:
        import fastwam  # noqa: F401
        return
    except ImportError:
        pass

    pkg_root = Path(__file__).resolve().parent / "FastWAM"
    src = pkg_root / "src"
    marker = src / "fastwam" / "__init__.py"
    if not marker.is_file():
        raise ImportError(
            "FastWAM upstream is missing. It is declared in .gitmodules under "
            "`policy/fastwam/FastWAM`. Fetch it with:\n"
            "  git submodule update --init --recursive\n"
            "then install:\n"
            "  pip install -e policy/fastwam/FastWAM\n"
            f"(expected {marker})"
        ) from None
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_setup_fastwam_import_path()


ATTENTION_MODE_REGISTRY = {
    "original": "fastwam.models.wan22.fastwam.FastWAM",
    "joint": "fastwam.models.wan22.fastwam_joint.FastWAMJoint",
    "idm": "fastwam.models.wan22.fastwam_idm.FastWAMIDM",
}


def _import_class(dotpath: str):
    module_path, cls_name = dotpath.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _resolve_dict(cfg):
    """Convert OmegaConf DictConfig to plain dict if needed."""
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg) if cfg is not None else {}


def _slice_prompts_for_data_parallel_batch(prompts: list, video_batch: int, video_device: torch.device) -> list:
    """Align prompt list with the local tensor batch under nn.DataParallel.

    ``scatter`` shards tensors on dim 0 but duplicates Python lists on every replica.
    ``encode_prompt`` must therefore see only the prompts for this replica's shard.
    """
    n = len(prompts)
    if n <= video_batch:
        return prompts
    if n % video_batch != 0:
        raise ValueError(
            f"raw_lang length ({n}) must match video batch ({video_batch}) "
            "or be an integer multiple when using DataParallel."
        )
    num_chunks = n // video_batch
    if num_chunks == 1:
        return prompts
    dev_idx = 0
    if video_device.type == "cuda":
        dev_idx = video_device.index if video_device.index is not None else 0
    elif torch.cuda.is_available():
        dev_idx = torch.cuda.current_device()
    return prompts[dev_idx * video_batch : (dev_idx + 1) * video_batch]


def _build_inner_wan_from_config(
    config: FastWAMPolicyConfig,
    *,
    device: str,
    torch_dtype: torch.dtype,
    skip_dit_load_from_pretrain: bool,
) -> nn.Module:
    """Instantiate upstream FastWAM (WAN + MoT) from a :class:`FastWAMPolicyConfig`."""
    if config.attention_mode not in ATTENTION_MODE_REGISTRY:
        raise ValueError(
            f"Unknown attention_mode '{config.attention_mode}'. "
            f"Must be one of: {list(ATTENTION_MODE_REGISTRY.keys())}"
        )
    if config.download_source is not None:
        os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = str(config.download_source)

    video_dit_config, action_dit_config = config.dit_configs_with_action_dim()
    video_dit_config = _resolve_dict(video_dit_config)
    action_dit_config = _resolve_dict(action_dit_config)
    video_scheduler = _resolve_dict(config.video_scheduler)
    action_scheduler = _resolve_dict(config.action_scheduler)
    loss = _resolve_dict(config.loss)

    if not video_dit_config:
        raise ValueError("`video_dit_config` is required to build FastWAM.")
    if not action_dit_config:
        raise ValueError("`action_dit_config` is required to build FastWAM.")

    model_cls = _import_class(ATTENTION_MODE_REGISTRY[config.attention_mode])
    logger.info(
        f"Building FastWAM model: attention_mode={config.attention_mode}, cls={model_cls.__name__}"
    )

    return model_cls.from_wan22_pretrained(
        device=device,
        torch_dtype=torch_dtype,
        model_id=config.model_id,
        tokenizer_model_id=config.tokenizer_model_id,
        tokenizer_max_len=int(config.tokenizer_max_len),
        load_text_encoder=bool(config.load_text_encoder),
        proprio_dim=None if config.proprio_dim is None else int(config.proprio_dim),
        redirect_common_files=bool(config.redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=config.action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(config.mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


class FastWAMPolicy(PreTrainedModel):
    """ILStudio policy wrapper around FastWAM world-action models.

    Uses :class:`FastWAMPolicyConfig` (HF ``PretrainedConfig``) like other ILStudio policies.
    """

    config_class = FastWAMPolicyConfig

    def __init__(self, config: FastWAMPolicyConfig):
        super().__init__(config)
        self._last_generated_video = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        skip_pretrain = bool(config.inference_mode) or bool(config.skip_dit_load_from_pretrain)
        self.model = _build_inner_wan_from_config(
            config,
            device=device,
            torch_dtype=torch.bfloat16,
            skip_dit_load_from_pretrain=skip_pretrain,
        )

    def reset(self):
        self._last_generated_video = None

    @property
    def device(self):
        return self.model.device

    @property
    def torch_dtype(self):
        return self.model.torch_dtype

    def forward(self, sample, **kwargs):
        """Training forward -- delegates to the underlying FastWAM training_loss.

        If the batch contains ``raw_lang`` instead of precomputed ``context``/
        ``context_mask``, encode the prompts on-the-fly using the loaded text
        encoder (requires ``load_text_encoder=true``).
        """
        if "context" not in sample and "raw_lang" in sample:
            prompts = sample.pop("raw_lang")
            if isinstance(prompts, str):
                prompts = [prompts]
            if "video" in sample:
                prompts = _slice_prompts_for_data_parallel_batch(
                    prompts, int(sample["video"].shape[0]), sample["video"].device
                )
            context, context_mask = self.model.encode_prompt(prompts)
            sample["context"] = context
            sample["context_mask"] = context_mask
        loss, loss_dict = self.model.training_loss(sample, **kwargs)
        # nn.DataParallel gathers only tensor leaves; training_loss uses Python floats for logging.
        loss_dict = {
            k: torch.as_tensor(v, device=loss.device, dtype=loss.dtype)
            for k, v in loss_dict.items()
        }
        return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch_obs: dict) -> torch.Tensor:
        """Inference interface required by ILStudio MetaPolicy.

        Args:
            batch_obs: dict with at least:
                - ``image``: ``[B, C, H, W]`` float tensor in [-1, 1]
                - ``raw_lang`` or ``context``/``context_mask``: language condition
                Optional:
                - ``state``: ``[B, state_dim]`` proprio
        Returns:
            ``[B, T, action_dim]`` action tensor (float32, cpu).
        """
        self.model.eval()

        image = batch_obs["image"]
        if image.ndim == 5:
            image = image[:, 0]
        B = image.shape[0]

        actions_list = []
        for i in range(B):
            input_image = image[i:i+1]

            prompt = None
            context = None
            context_mask = None
            if "context" in batch_obs and "context_mask" in batch_obs:
                ctx = batch_obs["context"]
                ctx_mask = batch_obs["context_mask"]
                context = ctx[i:i+1] if ctx.ndim == 3 else ctx
                context_mask = ctx_mask[i:i+1] if ctx_mask.ndim == 2 else ctx_mask
            elif "raw_lang" in batch_obs:
                lang = batch_obs["raw_lang"]
                prompt = lang[i] if isinstance(lang, (list, tuple)) else lang
            else:
                prompt = ""

            proprio = None
            if "state" in batch_obs and self.model.proprio_encoder is not None:
                state = batch_obs["state"]
                proprio = state[i:i+1] if state.ndim >= 2 else state.unsqueeze(0)

            infer_kwargs = dict(
                prompt=prompt,
                input_image=input_image,
                action_horizon=int(self.config.chunk_size),
                num_inference_steps=self.config.num_inference_steps,
                sigma_shift=self.config.sigma_shift,
                seed=self.config.seed,
                tiled=False,
            )
            if proprio is not None:
                infer_kwargs["proprio"] = proprio
            if context is not None:
                infer_kwargs["context"] = context
                infer_kwargs["context_mask"] = context_mask
                infer_kwargs["prompt"] = None

            needs_video_frames = (
                self.config.attention_mode in ("joint", "idm")
                or self.config.generate_video
            )
            if needs_video_frames:
                infer_kwargs["num_video_frames"] = self.config.num_video_frames

            if self.config.generate_video:
                infer_kwargs["test_action_with_infer_action"] = False
                pred = self.model.infer_joint(**infer_kwargs)
                self._last_generated_video = pred.get("video", None)
            else:
                pred = self.model.infer_action(**infer_kwargs)

            actions_list.append(pred["action"].unsqueeze(0))

        return torch.cat(actions_list, dim=0)

    def save_checkpoint(self, path, **kwargs):
        self.model.save_checkpoint(path, **kwargs)

    def load_checkpoint(self, path, **kwargs):
        return self.model.load_checkpoint(path, **kwargs)
