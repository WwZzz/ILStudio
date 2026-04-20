"""
DreamZero policy model for ILStudio.

Wrapper around the upstream DreamZero VLA (Video-Language-Action) model.
All core model code lives in the ``dreamzero`` git submodule under
``policy/dreamzero/dreamzero/groot/vla/model/dreamzero/``.

Supports loading pretrained DreamZero checkpoints (safetensors, Hydra output
dirs) as well as standard HuggingFace ``from_pretrained``.
"""

import importlib
import os
import gc
import json
import copy
import sys
import types
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from loguru import logger
from transformers import PretrainedConfig, PreTrainedModel


# ======================================================================
# Hydra shim — replaces hydra.utils.instantiate without the dependency
# ======================================================================
def _instantiate(cfg, *args, **kwargs):
    """Minimal drop-in replacement for ``hydra.utils.instantiate``.

    Resolves ``_target_`` in a nested dict config, imports the class,
    and calls it with the remaining keys as keyword arguments.
    Supports recursive ``_target_`` nesting.
    """
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return cfg
    cfg = dict(cfg)
    target = cfg.pop("_target_", None)
    cfg.pop("_convert_", None)
    cfg.pop("_recursive_", None)
    if target is None:
        return cfg

    resolved = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and "_target_" in v:
            resolved[k] = _instantiate(v)
        else:
            resolved[k] = v

    module_path, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(*args, **resolved, **kwargs)


def _install_hydra_shim() -> None:
    """Inject a fake ``hydra.utils`` module so upstream code that does
    ``from hydra.utils import instantiate`` works without installing hydra."""
    if "hydra" in sys.modules and hasattr(sys.modules["hydra"], "__file__"):
        return  # real hydra is already installed

    hydra_mod = types.ModuleType("hydra")
    hydra_mod.__path__ = []
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = _instantiate
    hydra_mod.utils = hydra_utils
    sys.modules.setdefault("hydra", hydra_mod)
    sys.modules.setdefault("hydra.utils", hydra_utils)


# ======================================================================
# Submodule import setup (mirrors policy/fastwam/modeling.py pattern)
# ======================================================================
def _setup_dreamzero_import_path() -> None:
    """Ensure the upstream ``groot`` package from the DreamZero submodule is importable.

    The upstream repo is a **git submodule** at ``policy/dreamzero/dreamzero/``.
    Prefer a local editable install::

        git submodule update --init --recursive
        pip install -e policy/dreamzero/dreamzero

    If the package is not installed, the submodule root is prepended to
    ``sys.path`` (development convenience only).
    """
    _install_hydra_shim()

    try:
        import groot.vla.model.dreamzero  # noqa: F401
        return
    except ImportError:
        pass

    pkg_root = Path(__file__).resolve().parent / "dreamzero"
    marker = pkg_root / "groot" / "vla" / "model" / "dreamzero" / "__init__.py"
    if not marker.is_file():
        raise ImportError(
            "DreamZero upstream is missing. It is declared in .gitmodules under "
            "`policy/dreamzero/dreamzero`. Fetch it with:\n"
            "  git submodule update --init --recursive\n"
            "then optionally install:\n"
            "  pip install -e policy/dreamzero/dreamzero\n"
            f"(expected {marker})"
        ) from None
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


_setup_dreamzero_import_path()

# The submodule decorates scheduler methods with ``@torch.compile(..., fullgraph=True)``
# which causes ``FailOnRecompileLimitHit`` at inference due to varying input shapes.
# Disable dynamo globally so ``@torch.compile`` becomes a pass-through.
import torch._dynamo
torch._dynamo.config.disable = True


def _patch_wan_policy_head_safe_device_dtype() -> None:
    """Upstream ``WANPolicyHead.device`` / ``.dtype`` use ``next(iter(self.parameters()))``.
    That raises ``StopIteration`` when the module tree has no ``Parameter`` s (seen with
    ``nn.DataParallel`` / VRAM wrappers). Patch from ILStudio without editing the submodule."""
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead

    def _first_tensor_leaf(mod: nn.Module):
        for p in mod.parameters(recurse=True):
            return p
        for b in mod.buffers(recurse=True):
            return b
        return None

    def _dtype(self):
        # LoRA parameters are explicitly float32 (line 600 upstream) while the
        # base model is bf16. Return model_dtype from config so that input
        # tensors are cast to the correct compute dtype, not the LoRA dtype.
        cfg = getattr(self, "config", None)
        md = getattr(cfg, "model_dtype", None)
        if md is not None:
            if isinstance(md, torch.dtype):
                return md
            name = str(md).replace("torch.", "").lower()
            dt = getattr(torch, name, None)
            if dt is not None:
                return dt
        t = _first_tensor_leaf(self)
        if t is not None:
            return t.dtype
        return torch.bfloat16

    def _device(self):
        t = _first_tensor_leaf(self)
        if t is not None:
            return t.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANPolicyHead.dtype = property(_dtype)  # type: ignore[assignment]
    WANPolicyHead.device = property(_device)  # type: ignore[assignment]


def _patch_wan_image_encoder_encode_image() -> None:
    """``WanImageEncoder.encode_image`` uses ``next(iter(self.model.visual.parameters()))``;
    under ``DataParallel`` that iterator can be empty → ``StopIteration``. ILStudio-only patch."""
    import torch.nn.functional as F

    from groot.vla.model.dreamzero.modules.wan_video_image_encoder import WanImageEncoder

    def _dtype_from_module(mod: nn.Module) -> torch.dtype:
        for p in mod.parameters(recurse=True):
            return p.dtype
        for b in mod.buffers(recurse=True):
            return b.dtype
        return torch.float32

    def encode_image_patched(self, videos):
        size = (self.model.image_size,) * 2
        videos = torch.cat(
            [
                F.interpolate(u, size=size, mode="bicubic", align_corners=False)
                for u in videos
            ]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))
        dtype = _dtype_from_module(self.model.visual)
        videos = videos.to(dtype)
        out = self.model.visual(videos, use_31_block=True)
        return out.clone()

    WanImageEncoder.encode_image = encode_image_patched  # type: ignore[assignment]


_patch_wan_policy_head_safe_device_dtype()
_patch_wan_image_encoder_encode_image()


def _patch_causal_rope_multi_chunk() -> None:
    """The upstream submodule's ``causal_rope_action_apply_polar`` (and its
    non-polar variant) assert ``action_register_length ==
    num_action_per_block + num_state_per_block``, which only allows
    ``chunk_size == 1`` (all actions/states in a single block).

    The dreamzero branch relaxed this to ``action_register_length %
    block_register_length == 0``, supporting multi-block layouts where
    ``num_action_per_block`` and ``num_state_per_block`` are *per temporal
    block* and the full horizon spans multiple blocks.

    We monkey-patch both functions here so that the unmodified submodule
    works with per-block action/state sizes used during training.
    """
    import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as _dit_mod

    _orig_polar = _dit_mod.causal_rope_action_apply_polar
    _orig_no_polar = _dit_mod.causal_rope_action_apply_no_polar

    def _patched_polar(x, freqs, freqs_action, freqs_state,
                       action_register_length, num_action_per_block,
                       num_state_per_block, action_state_index):
        import torch as _t
        B, seq_len, n, _ = x.shape
        x = _t.view_as_complex(
            x.to(_t.float64).reshape(B, seq_len, n, -1, 2)
        )
        if action_register_length is not None:
            blk = num_action_per_block + num_state_per_block
            assert action_register_length % blk == 0, (
                f"action_register_length={action_register_length} not divisible "
                f"by block_register_length={blk}"
            )
            chunk_size = action_register_length // blk
            if chunk_size == 1:
                fa = freqs_action[
                    action_state_index * num_action_per_block:
                    (action_state_index + 1) * num_action_per_block
                ]
                fs = freqs_state[
                    action_state_index * num_state_per_block:
                    (action_state_index + 1) * num_state_per_block
                ]
            else:
                fa = freqs_action[:chunk_size * num_action_per_block]
                fs = freqs_state[:chunk_size * num_state_per_block]
            freqs_1d = _t.cat([fa, fs], dim=0).view(action_register_length, 1, -1)
            freqs = _t.cat([freqs, freqs_1d], dim=0)
        freqs = freqs.unsqueeze(0)
        x = _t.view_as_real(x * freqs).flatten(3)
        return x

    def _patched_no_polar(x, freqs, freqs_action, freqs_state,
                          action_register_length, num_action_per_block,
                          num_state_per_block, action_state_index):
        import torch as _t
        B, seq_len, n, d = x.shape
        x = x.reshape(B, seq_len, n, -1, 2)
        x_real, x_imag = x[..., 0], x[..., 1]
        freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
        freqs_cos, freqs_sin = freqs[..., 0], freqs[..., 1]
        if action_register_length is not None:
            blk = num_action_per_block + num_state_per_block
            assert action_register_length % blk == 0
            chunk_size = action_register_length // blk
            if chunk_size == 1:
                fa = freqs_action[
                    action_state_index * num_action_per_block:
                    (action_state_index + 1) * num_action_per_block
                ]
                fs = freqs_state[
                    action_state_index * num_state_per_block:
                    (action_state_index + 1) * num_state_per_block
                ]
            else:
                fa = freqs_action[:chunk_size * num_action_per_block]
                fs = freqs_state[:chunk_size * num_state_per_block]
            freqs_1d = _t.cat([fa, fs], dim=0).view(
                action_register_length, 1, -1, 2
            )
            freqs_cos = _t.cat([freqs_cos[0], freqs_1d[..., 0]], dim=0).unsqueeze(0)
            freqs_sin = _t.cat([freqs_sin[0], freqs_1d[..., 1]], dim=0).unsqueeze(0)
        xr = x_real * freqs_cos - x_imag * freqs_sin
        xi = x_real * freqs_sin + x_imag * freqs_cos
        return _t.stack((xr, xi), dim=-1).flatten(3)

    _dit_mod.causal_rope_action_apply_polar = _patched_polar
    _dit_mod.causal_rope_action_apply_no_polar = _patched_no_polar
    _dit_mod.causal_rope_action_apply = lambda *a, **kw: (
        _patched_no_polar(*a, **kw) if _dit_mod.ENABLE_TENSORRT
        else _patched_polar(*a, **kw)
    )


_patch_causal_rope_multi_chunk()


def _silence_submodule_verbose() -> None:
    """Suppress per-step per-rank noise from the upstream submodule:
    - tqdm progress bars in VAE tiled encode/decode
    - ``print("videos", ...)`` inside WANPolicyHead.forward
    """
    import groot.vla.model.dreamzero.modules.wan_video_vae as _vae_mod
    _vae_mod.tqdm = lambda iterable, *a, **kw: iterable  # type: ignore[assignment]

    import builtins
    _original_print = builtins.print
    from groot.vla.model.dreamzero.action_head import wan_flow_matching_action_tf as _ah_mod

    def _quiet_print(*args, **kwargs):
        # Upstream: ``print("videos", videos.shape)`` — first arg is ``"videos"``, not ``"videos "``.
        if args and isinstance(args[0], str) and args[0] == "videos":
            return
        _original_print(*args, **kwargs)

    _ah_mod.print = _quiet_print  # type: ignore[assignment]


_silence_submodule_verbose()

from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig


# ======================================================================
# Config
# ======================================================================
class DreamZeroPolicyConfig(PretrainedConfig):
    model_type = "dreamzero"

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 8,
        action_horizon: int = 24,
        num_video_frames: int = 33,
        max_action_dim: int = 64,
        max_state_dim: int = 64,
        state_horizon: int = 16,
        num_inference_steps: int = 10,
        cfg_scale: float = 5.0,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        train_architecture: str = "lora",
        skip_component_loading: bool = False,
        use_gradient_checkpointing: bool = True,
        wan_model_path: str = "",
        vae_path: str = "",
        text_encoder_path: str = "",
        image_encoder_path: str = "",
        tokenizer_path: str = "google/umt5-xxl",
        embodiment_tag_mapping: dict = None,
        num_views: int = 2,
        image_size: list = None,
        tiled: bool = True,
        tile_size_height: int = 34,
        tile_size_width: int = 34,
        tile_stride_height: int = 18,
        tile_stride_width: int = 16,
        num_frame_per_block: int = 1,
        decouple_video_action_noise: bool = False,
        use_high_noise_emphasis: bool = False,
        compute_dtype: str = "bfloat16",
        dreamzero_pretrained_path: str = "",
        backbone_cfg: dict = None,
        action_head_cfg: dict = None,
        diffusion_model_cfg: dict = None,
        text_encoder_cfg: dict = None,
        image_encoder_cfg: dict = None,
        vae_cfg: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.num_video_frames = num_video_frames
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.state_horizon = state_horizon
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.train_architecture = train_architecture
        self.skip_component_loading = skip_component_loading
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.wan_model_path = wan_model_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.image_encoder_path = image_encoder_path
        self.tokenizer_path = tokenizer_path
        self.embodiment_tag_mapping = embodiment_tag_mapping or {}
        self.num_views = num_views
        self.image_size = image_size or [256, 256]
        self.tiled = tiled
        self.tile_size_height = tile_size_height
        self.tile_size_width = tile_size_width
        self.tile_stride_height = tile_stride_height
        self.tile_stride_width = tile_stride_width
        self.num_frame_per_block = num_frame_per_block
        self.decouple_video_action_noise = decouple_video_action_noise
        self.use_high_noise_emphasis = use_high_noise_emphasis
        self.compute_dtype = compute_dtype
        self.dreamzero_pretrained_path = dreamzero_pretrained_path
        self.backbone_cfg = backbone_cfg
        self.action_head_cfg = action_head_cfg
        self.diffusion_model_cfg = diffusion_model_cfg
        self.text_encoder_cfg = text_encoder_cfg
        self.image_encoder_cfg = image_encoder_cfg
        self.vae_cfg = vae_cfg


def _build_default_backbone_cfg():
    # Matches groot/vla/configs/model/dreamzero/backbone/identity.yaml — no nested config;
    # IdentityBackbone has no __init__ args beyond Backbone.
    return {
        "_target_": "groot.vla.model.dreamzero.backbone.identity.IdentityBackbone",
    }


def _build_default_action_head_cfg(config: DreamZeroPolicyConfig):
    num_latent_frames = (config.num_video_frames - 1) // 4 + 1
    num_blocks = (num_latent_frames - 1) // config.num_frame_per_block
    num_action_per_block = config.action_horizon // num_blocks
    num_state_per_block = config.state_horizon // num_blocks
    frame_seqlen = (config.image_size[0] // 16) * (config.image_size[1] // 16)

    ah_cfg = {
        "_target_": "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead",
        "config": {
            "_target_": "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHeadConfig",
            "model_dtype": config.compute_dtype,
            "action_dim": config.action_dim,
            "action_horizon": config.action_horizon,
            "max_action_dim": config.max_action_dim,
            "max_state_dim": config.max_state_dim,
            "num_frames": config.num_video_frames,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "train_architecture": config.train_architecture,
            "skip_component_loading": config.skip_component_loading,
            "use_gradient_checkpointing": config.use_gradient_checkpointing,
            "tiled": config.tiled,
            "tile_size_height": config.tile_size_height,
            "tile_size_width": config.tile_size_width,
            "tile_stride_height": config.tile_stride_height,
            "tile_stride_width": config.tile_stride_width,
            "num_frame_per_block": config.num_frame_per_block,
            "decouple_video_action_noise": config.decouple_video_action_noise,
            "use_high_noise_emphasis": config.use_high_noise_emphasis,
            "diffusion_model_cfg": config.diffusion_model_cfg or {
                "_target_": "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel",
                "model_type": "i2v",
                "dim": 5120,
                "in_dim": 36,
                "ffn_dim": 13824,
                "out_dim": 16,
                "freq_dim": 256,
                "eps": 1e-6,
                "num_heads": 40,
                "num_layers": 40,
                "num_frame_per_block": config.num_frame_per_block,
                "num_action_per_block": num_action_per_block,
                "num_state_per_block": num_state_per_block,
                "action_dim": config.max_action_dim,
                "max_state_dim": config.max_state_dim,
                "frame_seqlen": frame_seqlen,
            },
            "text_encoder_cfg": config.text_encoder_cfg or {
                "_target_": "groot.vla.model.dreamzero.modules.wan_video_text_encoder.WanTextEncoder",
                "text_encoder_pretrained_path": config.text_encoder_path or None,
            },
            "image_encoder_cfg": config.image_encoder_cfg or {
                "_target_": "groot.vla.model.dreamzero.modules.wan_video_image_encoder.WanImageEncoder",
                "image_encoder_pretrained_path": config.image_encoder_path or None,
            },
            "vae_cfg": config.vae_cfg or {
                "_target_": "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE",
                "vae_pretrained_path": config.vae_path or None,
            },
        },
    }
    return ah_cfg


# ======================================================================
# Pretrained weight loading utilities
# ======================================================================
def _load_safetensors_state_dict(checkpoint_dir):
    from safetensors.torch import load_file

    single = os.path.join(checkpoint_dir, "model.safetensors")
    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")

    state_dict = {}
    if os.path.exists(index):
        with open(index, "r") as f:
            idx = json.load(f)
        for shard in sorted(set(idx["weight_map"].values())):
            shard_path = os.path.join(checkpoint_dir, shard)
            logger.info(f"Loading shard: {shard_path}")
            shard_sd = load_file(shard_path)
            state_dict.update(shard_sd)
            del shard_sd
            gc.collect()
    elif os.path.exists(single):
        logger.info(f"Loading weights: {single}")
        state_dict = load_file(single)
    else:
        raise FileNotFoundError(
            f"No safetensors found in {checkpoint_dir}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )
    return state_dict


def _load_dreamzero_hydra_checkpoint(checkpoint_dir):
    """Load a DreamZero checkpoint saved via Hydra experiment pipeline.

    Expected layout::

        checkpoint_dir/
            config.json          (VLAConfig)
            model.safetensors*   (weights)
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        logger.info(f"Loaded DreamZero config from {config_path}")

    state_dict = _load_safetensors_state_dict(checkpoint_dir)
    return config_dict, state_dict


def _torch_dtype_from_compute_dtype(compute_dtype: str) -> torch.dtype:
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    name = str(compute_dtype).replace("torch.", "").lower()
    return getattr(torch, name, torch.bfloat16)


def _align_wan_action_head_dtypes(vla: nn.Module, compute_dtype: str) -> None:
    """Upstream WAN head casts ``videos`` to ``action_head.dtype`` (bf16) before VAE encode, but
    VAE / DiT weights may still be fp32 → conv3d ``Input type (BFloat16) and bias type (float)``.
    Cast heavy submodules to ``compute_dtype`` here (ILStudio only; submodule unchanged)."""
    if not hasattr(vla, "action_head"):
        return
    head = vla.action_head
    dt = _torch_dtype_from_compute_dtype(compute_dtype)
    for name in ("vae", "model"):
        sub = getattr(head, name, None)
        if sub is None:
            continue
        try:
            sub.to(dtype=dt)
        except Exception as exc:
            logger.warning(f"DreamZeroPolicy: could not cast action_head.{name} to {dt}: {exc}")


# ======================================================================
# Policy model
# ======================================================================
class DreamZeroPolicy(PreTrainedModel):
    config_class = DreamZeroPolicyConfig

    def __init__(self, config: DreamZeroPolicyConfig):
        super().__init__(config)

        backbone_cfg = config.backbone_cfg or _build_default_backbone_cfg()
        action_head_cfg = config.action_head_cfg or _build_default_action_head_cfg(config)

        vla_cfg = VLAConfig(
            backbone_cfg=backbone_cfg,
            action_head_cfg=action_head_cfg,
            action_horizon=config.action_horizon,
            action_dim=config.max_action_dim,
            compute_dtype=config.compute_dtype,
        )
        self.vla = VLA(vla_cfg)
        _align_wan_action_head_dtypes(self.vla, config.compute_dtype)

        self._frame_buffer = deque(maxlen=config.num_video_frames)
        self._state_buffer = deque(maxlen=config.state_horizon)

    @classmethod
    def from_dreamzero_checkpoint(
        cls,
        checkpoint_path: str,
        override_config: dict = None,
        load_lora: bool = False,
        device: str = "cuda",
    ):
        """Load a DreamZero model from its native checkpoint format."""
        config_dict, state_dict = _load_dreamzero_hydra_checkpoint(checkpoint_path)

        merged = {**config_dict, **(override_config or {})}
        policy_config = cls._translate_vla_config(merged)
        policy_config.skip_component_loading = True

        model = cls(policy_config)

        vla_keys = [k for k in state_dict if k.startswith("backbone.") or k.startswith("action_head.")]
        if vla_keys:
            missing, unexpected = model.vla.load_state_dict(state_dict, strict=False)
        else:
            stripped = {}
            for k, v in state_dict.items():
                stripped[k[4:] if k.startswith("vla.") else k] = v
            missing, unexpected = model.vla.load_state_dict(stripped, strict=False)

        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
        if not missing and not unexpected:
            logger.info("Successfully loaded all pretrained weights")

        _align_wan_action_head_dtypes(model.vla, model.config.compute_dtype)
        model.to(device)
        return model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Restore weights then re-apply compute dtype on VAE/DiT.

        HF ``from_pretrained`` runs ``_fix_key`` which renames every
        ``gamma`` → ``weight`` and ``beta`` → ``bias`` (legacy LayerNorm
        compat).  DreamZero's ``WanLayerNorm`` genuinely uses ``gamma``
        as a parameter name, so we reverse that renaming here.
        """
        model_keys = set(self.state_dict().keys())
        fixed = {}
        for k, v in state_dict.items():
            if k not in model_keys and ".weight" in k:
                alt = k.replace(".weight", ".gamma")
                if alt in model_keys:
                    k = alt
            if k not in model_keys and ".bias" in k:
                alt = k.replace(".bias", ".beta")
                if alt in model_keys:
                    k = alt
            fixed[k] = v
        try:
            incomp = super().load_state_dict(fixed, strict=strict, assign=assign)
        except TypeError:
            incomp = super().load_state_dict(fixed, strict=strict)
        _align_wan_action_head_dtypes(self.vla, self.config.compute_dtype)
        return incomp

    @staticmethod
    def _translate_vla_config(config_dict):
        ah_cfg = config_dict.get("action_head_cfg", {})
        ah_inner = ah_cfg.get("config", ah_cfg) if isinstance(ah_cfg, dict) else {}

        pc = DreamZeroPolicyConfig(
            action_dim=config_dict.get("action_dim", ah_inner.get("action_dim", 7)),
            action_horizon=config_dict.get("action_horizon", ah_inner.get("action_horizon", 24)),
            max_action_dim=ah_inner.get("max_action_dim", 64),
            max_state_dim=ah_inner.get("max_state_dim", 64),
            lora_rank=ah_inner.get("lora_rank", 4),
            lora_alpha=ah_inner.get("lora_alpha", 4),
            train_architecture=ah_inner.get("train_architecture", "lora"),
            use_gradient_checkpointing=ah_inner.get("use_gradient_checkpointing", True),
            tiled=ah_inner.get("tiled", True),
            tile_size_height=ah_inner.get("tile_size_height", 34),
            tile_size_width=ah_inner.get("tile_size_width", 34),
            compute_dtype=config_dict.get("compute_dtype", "bfloat16"),
            backbone_cfg=config_dict.get("backbone_cfg"),
            action_head_cfg=ah_cfg if ah_cfg else None,
        )
        return pc

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        from transformers.feature_extraction_utils import BatchFeature

        if isinstance(batch, BatchFeature):
            data = dict(batch.data)
        else:
            data = dict(batch)

        # WANPolicyHead asserts actions in [-1, 1]. ILStudio minmax can exceed that when
        # values fall outside dataset stats; bf16 can nudge 1.0 slightly over. Clamp here
        # (submodule unchanged).
        act = data.get("action")
        if isinstance(act, torch.Tensor) and torch.is_floating_point(act):
            data["action"] = torch.clamp(act, min=-1.0, max=1.0)

        # WANPolicyHead.forward (line 795): ``has_real_action[:, None] * action_loss``
        # where action_loss is 3-D ``(B, action_horizon, max_action_dim)``.
        # ``[:, None]`` on a 1-D ``(B,)`` gives ``(B, 1)`` which PyTorch broadcasts
        # by left-padding to ``(1, B, 1)`` → dim-1 mismatch with action_horizon.
        # The upstream code implicitly expects ``has_real_action`` to already be
        # 2-D ``(B, 1)`` so that ``[:, None]`` → ``(B, 1, 1)`` broadcasts correctly.
        hra = data.get("has_real_action")
        if isinstance(hra, torch.Tensor) and isinstance(act, torch.Tensor):
            B = int(act.shape[0])
            if hra.dim() == 0:
                hra = hra.expand(B).contiguous()
            elif hra.dim() >= 2:
                hra = hra.reshape(B, -1)
                if hra.shape[1] > 1:
                    hra = hra.any(dim=1) if hra.dtype == torch.bool else (hra != 0).any(dim=1)
                else:
                    hra = hra.squeeze(1)
            data["has_real_action"] = hra.view(B, 1)

        model_input = BatchFeature(data=data)
        outputs = self.vla(model_input)

        # VLA returns a BatchFeature, but nn.DataParallel.gather cannot
        # reconstruct BatchFeature from per-replica results. Convert to a
        # plain dict of tensors so gather can concatenate them normally.
        if isinstance(outputs, BatchFeature):
            outputs = dict(outputs.data)
        return outputs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def reset(self):
        self._frame_buffer.clear()
        self._state_buffer.clear()

    _NEG_PROMPT = (
        "Vibrant colors, overexposed, static, blurry details, text, "
        "subtitles, style, artwork, painting, image, still, grayscale, "
        "dull, worst quality, low quality, JPEG artifacts, ugly."
    )

    def _ensure_cfg_text(self, data: dict, device) -> dict:
        """Ensure ``text_negative`` / ``text_attention_mask_negative`` exist
        so that ``lazy_joint_video_action`` can do classifier-free guidance
        when ``cfg_scale != 1``."""
        if "text_negative" in data:
            return data

        from transformers import AutoTokenizer

        if not hasattr(self, "_inf_tokenizer") or self._inf_tokenizer is None:
            self._inf_tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path, local_files_only=True
            )

        def _tokenize(texts):
            enc = self._inf_tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.config.max_text_len,
                add_special_tokens=True,
            )
            return enc.input_ids.to(device), enc.attention_mask.to(device)

        if "text" not in data:
            raw = data.get("raw_lang", "")
            if isinstance(raw, (list, tuple)):
                raw = [str(r) for r in raw]
            else:
                raw = [str(raw)]
            ids, mask = _tokenize(raw)
            data = {**data, "text": ids, "text_attention_mask": mask}

        batch_size = data["text"].shape[0] if isinstance(data["text"], torch.Tensor) else 1
        neg_ids, neg_mask = _tokenize([self._NEG_PROMPT] * batch_size)
        data = {**data, "text_negative": neg_ids, "text_attention_mask_negative": neg_mask}
        return data

    @torch.no_grad()
    def select_action(self, batch_obs):
        """Run inference for ILStudio's MetaPolicy.

        Parameters
        ----------
        batch_obs : dict
            image  – (B, K, C, H, W) or (K, C, H, W)
            qpos / state – (B, state_dim) or (state_dim,)
            raw_lang – str or list[str]

        Returns
        -------
        actions : torch.Tensor  (B, action_horizon, action_dim)
        """
        from transformers.feature_extraction_utils import BatchFeature

        self.vla.eval()
        device = next(self.parameters()).device

        if isinstance(batch_obs, dict) and "images" in batch_obs and "text" in batch_obs:
            images = batch_obs["images"]
            if isinstance(images, torch.Tensor) and images.dim() == 5 and images.shape[-1] in (1, 3):
                tgt_h, tgt_w = self.config.image_size
                _, T, H, W, C = images.shape
                if (H, W) != (tgt_h, tgt_w):
                    images = images.permute(0, 1, 4, 2, 3).flatten(0, 1)
                    images = torch.nn.functional.interpolate(
                        images.float(), size=(tgt_h, tgt_w), mode="bilinear", align_corners=False,
                    ).to(dtype=images.dtype)
                    images = images.unflatten(0, (-1, T)).permute(0, 1, 3, 4, 2)
                    batch_obs = {**batch_obs, "images": images}

                batch_obs = self._ensure_cfg_text(batch_obs, device)
                model_input = BatchFeature(
                    data={
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch_obs.items()
                    }
                )
                outputs = self.vla.lazy_joint_video_action(model_input)
                pred_actions = outputs["action_pred"]
                return pred_actions[:, :, : self.config.action_dim]

        image = batch_obs.get("image", batch_obs.get("images"))
        state = batch_obs.get("qpos", batch_obs.get("state"))
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif isinstance(image, torch.Tensor) and image.dim() == 4:
            image = image.unsqueeze(0)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            state = state.unsqueeze(0)

        if isinstance(image, torch.Tensor):
            frame = image.permute(0, 1, 3, 4, 2).cpu().numpy()
            self._frame_buffer.append(frame[0])

        if isinstance(state, torch.Tensor):
            self._state_buffer.append(state[0].cpu().numpy())

        num_frames = self.config.num_video_frames
        frames = list(self._frame_buffer)
        if len(frames) < num_frames:
            padding = [frames[0]] * (num_frames - len(frames))
            frames = padding + frames
        frames = frames[-num_frames:]
        video = np.stack(frames, axis=0)

        if video.shape[1] > 1:
            cams = [video[:, i] for i in range(video.shape[1])]
            video = np.concatenate(cams, axis=2)
        else:
            video = video[:, 0]
        video = video[np.newaxis]

        states = list(self._state_buffer)
        sh = self.config.state_horizon
        if len(states) < sh:
            states = [states[0]] * (sh - len(states)) + states
        states = states[-sh:]
        state_arr = np.stack(states, axis=0)
        if state_arr.shape[-1] < self.config.max_state_dim:
            pad_w = self.config.max_state_dim - state_arr.shape[-1]
            state_arr = np.pad(state_arr, ((0, 0), (0, pad_w)))
        state_tensor = torch.from_numpy(state_arr).float().unsqueeze(0).to(device)

        video_tensor = torch.from_numpy(video).to(device)
        action_placeholder = torch.zeros(
            1, self.config.action_horizon, self.config.max_action_dim, device=device
        )
        action_mask = torch.zeros_like(action_placeholder, dtype=torch.bool)
        action_mask[:, :, : self.config.action_dim] = True

        raw_lang = batch_obs.get("raw_lang", "")

        inputs = {
            "images": video_tensor,
            "state": state_tensor,
            "action": action_placeholder,
            "action_mask": action_mask,
            "has_real_action": torch.ones(1, dtype=torch.bool, device=device),
            "embodiment_id": torch.zeros(1, dtype=torch.long, device=device),
            "raw_lang": raw_lang,
        }
        inputs = self._ensure_cfg_text(inputs, device)

        model_input = BatchFeature(data={
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        })

        outputs = self.vla.lazy_joint_video_action(model_input)
        pred_actions = outputs["action_pred"]

        return pred_actions[:, :, : self.config.action_dim]
