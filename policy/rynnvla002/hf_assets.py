"""Optional Hugging Face downloads for RynnVLA-002 checkpoints (README Step 0)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

# Upstream README Step 0: Alibaba-DAMO-Academy/WorldVLA
WORLDVLA_REPO = "Alibaba-DAMO-Academy/WorldVLA"
# Lumina text tokenizer (small files only)
LUMINA_REPO = "Alpha-VLLM/Lumina-mGPT-7B-768"


def _offline_mode() -> bool:
    v = os.environ.get("HF_HUB_OFFLINE", "").lower()
    if v in ("1", "true", "yes"):
        return True
    v = os.environ.get("TRANSFORMERS_OFFLINE", "").lower()
    if v in ("1", "true", "yes"):
        return True
    v = os.environ.get("ILSTUDIO_RYNN_NO_AUTO_DOWNLOAD", "").lower()
    if v in ("1", "true", "yes"):
        return True
    return False


def is_likely_hf_repo_id(s: str) -> bool:
    """True for ``org/name`` style ids that are not obvious filesystem paths."""
    s = (s or "").strip()
    if "/" not in s or s.count("/") != 1:
        return False
    if os.path.isabs(s) or s.startswith("./") or s.startswith("../"):
        return False
    org, name = s.split("/", 1)
    if not org or not name:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
    return all(c in allowed for c in org) and all(c in allowed for c in name)


def _chameleon_tokenizer_ready(d: Path) -> bool:
    return (d / "vqgan.yaml").is_file() and (d / "text_tokenizer.json").is_file()


def _starting_point_ready(d: Path) -> bool:
    if not d.is_dir():
        return False
    if (d / "config.json").is_file():
        return True
    if (d / "model.safetensors.index.json").is_file():
        return True
    return any(d.glob("*.safetensors")) or any(d.glob("pytorch_model*.bin"))


def _lumina_tokenizer_ready(d: Path) -> bool:
    return (d / "tokenizer.json").is_file() and (d / "tokenizer_config.json").is_file()


def _lumina_resolved_any(ckpts: Path) -> bool:
    base = ckpts / "models--Alpha-VLLM--Lumina-mGPT-7B-768" / "snapshots"
    if not base.is_dir():
        return False
    for snap in base.iterdir():
        if snap.is_dir() and _lumina_tokenizer_ready(snap):
            return True
    return False


def _copytree_merge(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def maybe_download_rynnvla_ckpts(*, policy_dir: Path, config: Any) -> None:
    """Populate ``RynnVLA-002/rynnvla-002/ckpts`` from Hugging Face when files are missing.

    Sources match the upstream RynnVLA-002 README (WorldVLA + Lumina-mGPT tokenizer).
    """
    if _offline_mode():
        logger.debug("Skipping RynnVLA-002 auto-download (offline / ILSTUDIO_RYNN_NO_AUTO_DOWNLOAD).")
        return
    if not getattr(config, "auto_download_ckpts", True):
        return

    ckpts = policy_dir / "RynnVLA-002" / "rynnvla-002" / "ckpts"
    if not ckpts.parent.is_dir():
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub is not available; skip RynnVLA-002 auto-download.")
        return

    tok_dst = ckpts / "chameleon" / "tokenizer"
    start_dst = ckpts / "starting_point"
    pp = (getattr(config, "pretrained_path", None) or "").strip()
    skip_starting_dl = is_likely_hf_repo_id(pp)

    need_tok = not _chameleon_tokenizer_ready(tok_dst)
    need_start = (not skip_starting_dl) and (not _starting_point_ready(start_dst))
    need_lumina = not _lumina_resolved_any(ckpts)

    if not need_tok and not need_start and not need_lumina:
        return

    if need_tok or need_start:
        patterns: list[str] = []
        if need_tok:
            patterns.append("chameleon/tokenizer/*")
        if need_start:
            patterns.append("chameleon/starting_point/*")
        staging = ckpts / "_hf_stage_WorldVLA"
        try:
            logger.info(
                "Auto-downloading Chameleon assets from {} (tokenizer / starting_point; first run may take a while)...",
                WORLDVLA_REPO,
            )
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=WORLDVLA_REPO,
                local_dir=str(staging),
                allow_patterns=patterns,
                local_dir_use_symlinks=False,
            )
            src_tok = staging / "chameleon" / "tokenizer"
            src_sp = staging / "chameleon" / "starting_point"
            if need_tok and _chameleon_tokenizer_ready(src_tok):
                _copytree_merge(src_tok, tok_dst)
            elif need_tok:
                logger.warning("WorldVLA snapshot missing chameleon/tokenizer; check network or HF access.")
            if need_start and _starting_point_ready(src_sp):
                _copytree_merge(src_sp, start_dst)
            elif need_start:
                logger.warning("WorldVLA snapshot missing chameleon/starting_point; check network or HF access.")
        except Exception as e:
            logger.error("WorldVLA auto-download failed: {}", e)
            raise RuntimeError(
                "自动下载 WorldVLA（Chameleon tokenizer / starting_point）失败。可设置环境变量 "
                "ILSTUDIO_RYNN_NO_AUTO_DOWNLOAD=1 后按 policy/rynnvla002/README.md 手动放置权重。"
            ) from e
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    if need_lumina and not _lumina_resolved_any(ckpts):
        lumina_dst = (
            ckpts / "models--Alpha-VLLM--Lumina-mGPT-7B-768" / "snapshots" / "hf_hub_auto"
        )
        try:
            logger.info("Auto-downloading Lumina-mGPT tokenizer files from {} ...", LUMINA_REPO)
            lumina_dst.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=LUMINA_REPO,
                local_dir=str(lumina_dst),
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.error("Lumina tokenizer auto-download failed: {}", e)
            raise RuntimeError(
                "自动下载 Lumina-mGPT tokenizer 失败。可设置 ILSTUDIO_RYNN_NO_AUTO_DOWNLOAD=1 "
                "后按 README 手动准备 models--Alpha-VLLM--Lumina-mGPT-7B-768。"
            ) from e
