import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

FASTWAM_ROOT = Path(__file__).resolve().parents[2] / "FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"
if str(FASTWAM_SRC) not in sys.path:
    sys.path.insert(0, str(FASTWAM_SRC))


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


class FastWAMPolicy(nn.Module):
    """ILStudio policy wrapper around FastWAM world-action models.

    Wraps any FastWAM variant (original / joint / idm) and exposes the
    ``select_action`` interface required by ILStudio's MetaPolicy.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_mode: str = "original",
        generate_video: bool = False,
        action_horizon: int = 24,
        num_video_frames: int = 17,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.attention_mode = attention_mode
        self.generate_video = generate_video
        self.action_horizon = action_horizon
        self.num_video_frames = num_video_frames
        self.num_inference_steps = num_inference_steps
        self.sigma_shift = sigma_shift
        self.seed = seed

        self._last_generated_video = None

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
            context, context_mask = self.model.encode_prompt(prompts)
            sample["context"] = context
            sample["context_mask"] = context_mask
        return self.model.training_loss(sample, **kwargs)

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
                action_horizon=self.action_horizon,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                tiled=False,
            )
            if proprio is not None:
                infer_kwargs["proprio"] = proprio
            if context is not None:
                infer_kwargs["context"] = context
                infer_kwargs["context_mask"] = context_mask
                infer_kwargs["prompt"] = None

            needs_video_frames = (
                self.attention_mode in ("joint", "idm")
                or self.generate_video
            )
            if needs_video_frames:
                infer_kwargs["num_video_frames"] = self.num_video_frames

            if self.generate_video:
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


def build_fastwam_model(
    attention_mode: str = "original",
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: Optional[int] = None,
    redirect_common_files: bool = True,
    mot_checkpoint_mixed_attn: bool = True,
    action_dit_pretrained_path: Optional[str] = None,
    skip_dit_load_from_pretrain: bool = False,
    video_dit_config: Optional[dict] = None,
    action_dit_config: Optional[dict] = None,
    video_scheduler: Optional[dict] = None,
    action_scheduler: Optional[dict] = None,
    loss: Optional[dict] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    generate_video: bool = False,
    action_horizon: int = 24,
    num_video_frames: int = 17,
    num_inference_steps: int = 20,
    sigma_shift: Optional[float] = None,
    seed: Optional[int] = None,
    download_source: Optional[str] = None,
) -> "FastWAMPolicy":
    """Build a FastWAMPolicy from scratch, downloading WAN weights automatically."""
    if attention_mode not in ATTENTION_MODE_REGISTRY:
        raise ValueError(
            f"Unknown attention_mode '{attention_mode}'. "
            f"Must be one of: {list(ATTENTION_MODE_REGISTRY.keys())}"
        )

    if download_source is not None:
        os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = download_source

    video_dit_config = _resolve_dict(video_dit_config)
    action_dit_config = _resolve_dict(action_dit_config)
    video_scheduler = _resolve_dict(video_scheduler)
    action_scheduler = _resolve_dict(action_scheduler)
    loss = _resolve_dict(loss)

    if not video_dit_config:
        raise ValueError("`video_dit_config` is required to build FastWAM.")
    if not action_dit_config:
        raise ValueError("`action_dit_config` is required to build FastWAM.")

    model_cls = _import_class(ATTENTION_MODE_REGISTRY[attention_mode])
    logger.info(f"Building FastWAM model: attention_mode={attention_mode}, cls={model_cls.__name__}")

    inner_model = model_cls.from_wan22_pretrained(
        device=device,
        torch_dtype=torch_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )

    policy = FastWAMPolicy(
        model=inner_model,
        attention_mode=attention_mode,
        generate_video=generate_video,
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
    )
    return policy
