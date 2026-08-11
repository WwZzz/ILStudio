"""Training-state and replay checkpoints for the RL runtime."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from rl.buffer import BaseBuffer


RL_STATE_FILENAME = "rl_state.pt"
REPLAY_STATE_FILENAME = "replay_buffer.pt"
_RL_STATE_FORMAT = "ilstudio.rl_state"
_REPLAY_STATE_FORMAT = "ilstudio.replay_buffer"
_STATE_VERSION = 1


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with torch releases predating ``weights_only``.
        return torch.load(path, map_location="cpu")


def _resolve_file(path, filename: str) -> Path:
    result = Path(path).expanduser()
    if result.is_dir():
        result = result / filename
    return result


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint rng state must be a mapping")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(cuda_state) != device_count:
            raise ValueError(
                "checkpoint CUDA RNG state has "
                f"{len(cuda_state)} devices, but this process has {device_count}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _buffer_class_name(buffer: BaseBuffer) -> str:
    return _class_name(buffer)


def _class_name(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _runtime_manifest(runner) -> dict[str, str]:
    return {
        "runner": _class_name(runner),
        "algorithm": _class_name(runner.algorithm),
        "buffer": _class_name(runner.buffer),
        "policy_adapter": _class_name(runner.policy_adapter),
        "trainer_adapter": _class_name(runner.trainer_adapter),
    }


def _validate_runtime_manifest(runner, manifest) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("RL checkpoint runtime manifest must be a mapping")
    expected = _runtime_manifest(runner)
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: {saved!r} != {current!r}"
            for key, (saved, current) in mismatches.items()
        )
        raise ValueError(f"RL checkpoint component mismatch ({detail})")


def save_replay_checkpoint(buffer: BaseBuffer, path) -> dict[str, Any]:
    """Write a reusable replay snapshot without coupling it to policy weights."""

    if not isinstance(buffer, BaseBuffer):
        raise TypeError("replay checkpoint requires a BaseBuffer")
    if buffer.buffer_type != "replay":
        raise ValueError("only replay buffers can be saved as replay checkpoints")
    path = _resolve_file(path, REPLAY_STATE_FILENAME).resolve()
    manifest = {
        "format": _REPLAY_STATE_FORMAT,
        "version": _STATE_VERSION,
        "buffer_class": _buffer_class_name(buffer),
        "buffer_type": buffer.buffer_type,
        "num_items": len(buffer),
        "num_env_steps": buffer.num_env_steps,
    }
    payload = {**manifest, "buffer": buffer.checkpoint_state_dict()}
    _atomic_torch_save(payload, path)
    return {**manifest, "path": str(path)}


def load_replay_checkpoint(buffer: BaseBuffer, path) -> dict[str, Any]:
    """Restore an independently saved replay snapshot into ``buffer``."""

    if not isinstance(buffer, BaseBuffer):
        raise TypeError("replay checkpoint requires a BaseBuffer")
    path = _resolve_file(path, REPLAY_STATE_FILENAME).resolve()
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError("replay checkpoint must contain a mapping")
    if payload.get("format") != _REPLAY_STATE_FORMAT:
        raise ValueError("unsupported replay checkpoint format")
    if payload.get("version") != _STATE_VERSION:
        raise ValueError("unsupported replay checkpoint version")
    expected_class = _buffer_class_name(buffer)
    if payload.get("buffer_class") != expected_class:
        raise ValueError(
            "replay buffer class mismatch: "
            f"{payload.get('buffer_class')} != {expected_class}"
        )
    if payload.get("buffer_type") != buffer.buffer_type:
        raise ValueError("replay buffer type mismatch")
    buffer.load_checkpoint_state_dict(payload["buffer"])
    if len(buffer) != payload.get("num_items"):
        raise RuntimeError("restored replay item count does not match its manifest")
    if buffer.num_env_steps != payload.get("num_env_steps"):
        raise RuntimeError(
            "restored replay environment-step count does not match its manifest"
        )
    return {
        key: value
        for key, value in payload.items()
        if key != "buffer"
    } | {"path": str(path)}


def save_rl_checkpoint(runner, output_dir, *, replay_path=None) -> Path:
    """Save policy files plus resumable non-replay state to ``rl_state.pt``."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_policy(output_dir)
    replay_manifest = None
    if replay_path is not None:
        replay_manifest = save_replay_checkpoint(runner.buffer, replay_path)
        try:
            replay_manifest["path"] = os.path.relpath(
                replay_manifest["path"], output_dir
            )
        except ValueError:
            # Windows cannot make a relative path across drive letters.
            replay_manifest["path"] = str(Path(replay_manifest["path"]))
    payload = {
        "format": _RL_STATE_FORMAT,
        "version": _STATE_VERSION,
        "runtime": _runtime_manifest(runner),
        "runner": runner.state_dict(include_buffer=False),
        "rng": _rng_state(),
        "replay": replay_manifest,
    }
    state_path = output_dir / RL_STATE_FILENAME
    _atomic_torch_save(payload, state_path)
    return state_path


def load_rl_checkpoint(
    runner,
    path,
    *,
    replay_path: Optional[str] = None,
) -> dict[str, Any]:
    """Restore runner state and optionally an independent replay snapshot.

    Passing ``replay_path="auto"`` follows the replay reference recorded in
    ``rl_state.pt``. Omitting it intentionally resumes with the newly built
    buffer, which avoids an unexpected large read.
    """

    state_path = _resolve_file(path, RL_STATE_FILENAME).resolve()
    payload = _torch_load(state_path)
    if not isinstance(payload, Mapping):
        raise TypeError("RL checkpoint must contain a mapping")
    if payload.get("format") != _RL_STATE_FORMAT:
        raise ValueError("unsupported RL checkpoint format")
    if payload.get("version") != _STATE_VERSION:
        raise ValueError("unsupported RL checkpoint version")
    _validate_runtime_manifest(runner, payload.get("runtime"))

    selected_replay = replay_path
    if selected_replay == "auto":
        manifest = payload.get("replay")
        if not isinstance(manifest, Mapping) or not manifest.get("path"):
            raise ValueError("rl_state.pt does not reference a replay checkpoint")
        selected_replay = state_path.parent / manifest["path"]

    runner.load_state_dict(payload["runner"], load_buffer=False)
    replay_manifest = None
    if selected_replay is not None:
        replay_manifest = load_replay_checkpoint(runner.buffer, selected_replay)
    _restore_rng_state(payload["rng"])
    return {
        "state_path": str(state_path),
        "replay": replay_manifest,
        "iteration": runner.iteration,
        "global_env_steps": runner.global_env_steps,
        "global_update_steps": runner.global_update_steps,
    }


__all__ = [
    "REPLAY_STATE_FILENAME",
    "RL_STATE_FILENAME",
    "load_replay_checkpoint",
    "load_rl_checkpoint",
    "save_replay_checkpoint",
    "save_rl_checkpoint",
]
