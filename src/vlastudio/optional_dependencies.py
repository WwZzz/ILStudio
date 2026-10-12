"""Actionable checks for VLAStudio's optional dependency layers.

Keep this module core-only: it is imported by the lightweight public API.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import sys
from typing import Iterable

import yaml


@dataclass(frozen=True)
class DependencyGroup:
    extra: str
    modules: tuple[str, ...]
    description: str


DATA_MODULES = ("numpy", "h5py", "PIL", "cv2", "imageio", "scipy")
TORCH_MODULES = ("torch", "torchvision")
POLICY_BASE_MODULES = ("numpy", "torch", "torchvision")


GROUPS = {
    "data": DependencyGroup(
        "data", DATA_MODULES,
        "shared dataset loading and preprocessing",
    ),
    "torch": DependencyGroup(
        "torch", TORCH_MODULES, "the supported PyTorch base",
    ),
    "act": DependencyGroup(
        "act", POLICY_BASE_MODULES + ("transformers", "accelerate", "einops", "timm"),
        "ACT policies",
    ),
    "diffusion": DependencyGroup(
        "diffusion", POLICY_BASE_MODULES + ("diffusers", "transformers", "einops", "timm"),
        "diffusion policies",
    ),
    "fastwam": DependencyGroup(
        "fastwam", POLICY_BASE_MODULES + ("hydra", "omegaconf", "huggingface_hub"),
        "FastWAM policies",
    ),
    "smolvla": DependencyGroup(
        "smolvla", POLICY_BASE_MODULES + ("transformers", "accelerate", "peft", "diffusers", "lerobot"),
        "SmolVLA policies",
    ),
    "openpi": DependencyGroup(
        "openpi", POLICY_BASE_MODULES + ("jax", "flax", "transformers"), "OpenPI policies",
    ),
    "qwen": DependencyGroup(
        "qwen", POLICY_BASE_MODULES + ("transformers", "accelerate", "peft", "diffusers"),
        "Qwen policy families",
    ),
    "openvla": DependencyGroup(
        "openvla", POLICY_BASE_MODULES + ("transformers", "accelerate", "peft", "bitsandbytes"),
        "OpenVLA policy families",
    ),
    "xvla": DependencyGroup(
        "xvla", POLICY_BASE_MODULES + ("transformers", "accelerate", "safetensors", "draccus"),
        "X-VLA policies",
    ),
    "roboflamingo": DependencyGroup(
        "roboflamingo", POLICY_BASE_MODULES + ("transformers", "einops_exts", "open_clip"),
        "RoboFlamingo policies",
    ),
    "octo": DependencyGroup(
        "octo", POLICY_BASE_MODULES + ("jax", "flax", "tensorflow", "tensorflow_datasets"),
        "Octo policies",
    ),
    "groot": DependencyGroup(
        "groot", POLICY_BASE_MODULES + ("transformers", "safetensors", "einops", "tree"),
        "GR00T policies",
    ),
}


POLICY_GROUPS = {
    "act": "act",
    "diffusion_policy": "diffusion",
    "diffusion_timm": "diffusion",
    "fastwam": "fastwam",
    "smolvla": "smolvla",
    "openpi": "openpi",
    "qwen_oft": "qwen",
    "qwen2dp": "qwen",
    "qwen25dp": "qwen",
    "florence2_oft": "qwen",
    "llava_oft": "qwen",
    "divla": "qwen",
    "openvla": "openvla",
    "openvla_oft": "openvla",
    "xvla": "xvla",
    "roboflamingo": "roboflamingo",
    "octo": "octo",
    "groot": "groot",
    "mlp": "act",
    "fm": "act",
    # Replay only needs the shared data/PyTorch layers for import.
    "replay": None,
}


class OptionalDependencyError(ImportError):
    """Raised with copy/paste installation commands for a selected component."""


def policy_component(config_path: str | Path) -> str | None:
    """Return the built-in component named by a policy YAML without importing it."""
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    target = str(config.get("type", ""))
    prefix = "vlastudio.policy."
    if not target.startswith(prefix):
        return None
    return target[len(prefix):].split(".", 1)[0]


def groups_for_policy(config_path: str | Path) -> tuple[str, ...]:
    component = policy_component(config_path)
    if component is None:
        return ()
    model_group = POLICY_GROUPS.get(component, component if component in GROUPS else None)
    return (model_group,) if model_group else ("data", "torch")


def _missing(groups: Iterable[str]) -> list[str]:
    missing = []
    for name in groups:
        for module in GROUPS[name].modules:
            try:
                available = find_spec(module) is not None
            except (ImportError, AttributeError, ValueError):
                available = False
            if not available and module not in missing:
                missing.append(module)
    return missing


def require_optional(groups: Iterable[str], *, operation: str) -> None:
    """Validate dependency groups and raise one precise, actionable error."""
    selected = tuple(dict.fromkeys(groups))
    if not selected:
        return
    missing = _missing(selected)
    if not missing:
        return
    extras = ",".join(GROUPS[name].extra for name in selected)
    executable = Path(sys.executable).name or "python"
    raise OptionalDependencyError(
        f"{operation} requires optional dependencies that are not installed: "
        f"{', '.join(missing)}.\n"
        f"Install the required VLAStudio layers with:\n"
        f"  {executable} -m pip install \"vlastudio[{extras}]\"\n"
        f"From a source checkout, use:\n"
        f"  {executable} -m pip install -e \".[{extras}]\"\n"
        "Then retry the same call. To let a managed VLAStudio worker provide the "
        "dependencies instead, pass check_dependencies=False."
    )
