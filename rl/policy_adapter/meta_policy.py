"""Adapter for existing ``MetaPolicy`` checkpoints."""

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import numpy as np
import torch

from benchmark.base import MetaAction, MetaObs, MetaPolicy, dict2meta
from rl.base import MetaTransition

from .base import BasePolicyAdapter


def _model_device(model):
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return next(parameters()).device
    except StopIteration:
        return None


def _move_to_device(value, device):
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


class MetaPolicyAdapter(BasePolicyAdapter):
    """Reuse ``MetaPolicy`` inference and native supervised policy forwards."""

    def __init__(
        self,
        meta_policy: MetaPolicy,
        *,
        exploration_std: float = 0.0,
        exploration_clip=None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        checkpoint_path=None,
    ) -> None:
        if not isinstance(meta_policy, MetaPolicy):
            raise TypeError("meta_policy must be a MetaPolicy")
        if exploration_std < 0:
            raise ValueError("exploration_std cannot be negative")
        if exploration_clip is not None:
            exploration_clip = tuple(exploration_clip)
            if len(exploration_clip) != 2 or exploration_clip[0] >= exploration_clip[1]:
                raise ValueError("exploration_clip must be (low, high)")

        model = meta_policy.policy
        super().__init__(
            model,
            capabilities=("action", "chunk_training"),
        )
        self.meta_policy = meta_policy
        self.exploration_std = float(exploration_std)
        self.exploration_clip = exploration_clip
        self.chunk_size = self._resolve_chunk_size(chunk_size)
        self._rng = np.random.default_rng(seed)
        self.checkpoint_path = checkpoint_path

    def _resolve_chunk_size(self, chunk_size):
        if chunk_size is not None:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer")
            return chunk_size

        config = getattr(self.policy, "config", SimpleNamespace())
        for name in ("chunk_size", "num_queries", "prediction_horizon"):
            value = getattr(config, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise ValueError("chunk_size is absent from both adapter arguments and policy config")

    @staticmethod
    def _single_step_meta_action(entry):
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            values = entry.reshape(-1).tolist()
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, Mapping)):
            values = list(entry)
        else:
            values = [entry]
        if len(values) != 1:
            raise ValueError("MetaPolicyAdapter only supports one synchronous environment")
        value = values[0]
        if isinstance(value, MetaAction):
            return value
        if isinstance(value, Mapping):
            return dict2meta(dict(value), mtype="action")
        raise TypeError("MetaPolicy inference entries must contain MetaAction mappings")

    def _decode_action_chunk(self, entries):
        if not isinstance(entries, (list, tuple)) or not entries:
            raise TypeError("MetaPolicy.inference must return a non-empty action list")
        steps = tuple(self._single_step_meta_action(entry) for entry in entries)
        first = steps[0]
        for step in steps:
            if step.ctrl_space != first.ctrl_space or step.ctrl_type != first.ctrl_type:
                raise ValueError("action chunk contains inconsistent control metadata")
        actions = [np.asarray(step.action) for step in steps]
        if any(action.ndim != 1 for action in actions):
            raise ValueError("each action-list entry must contain one action vector")
        return MetaAction(
            action=np.stack(actions).astype(np.float32, copy=False),
            ctrl_space=first.ctrl_space,
            ctrl_type=first.ctrl_type,
            gripper_continuous=first.gripper_continuous,
        )

    def _explore(self, action: MetaAction, std: float):
        if std == 0:
            return action
        normalizer = self.meta_policy.action_normalizer
        normalized = normalizer.normalize(
            np.asarray(action.action).copy(),
            datatype="action",
        )
        normalized = normalized + self._rng.normal(
            0.0,
            std,
            size=normalized.shape,
        ).astype(normalized.dtype)
        if self.exploration_clip is not None:
            normalized = np.clip(normalized, *self.exploration_clip)
        explored = normalizer.denormalize(normalized, datatype="action")
        return MetaAction(
            action=np.asarray(explored, dtype=np.float32),
            ctrl_space=action.ctrl_space,
            ctrl_type=action.ctrl_type,
            gripper_continuous=action.gripper_continuous,
        )

    def select_action(self, obs: MetaObs, *, deterministic=False, context=None):
        self._validate_obs(obs)
        with torch.inference_mode():
            entries = self.meta_policy.inference(copy.deepcopy(obs))
        action = self._decode_action_chunk(entries)
        std = 0.0 if deterministic else float(
            dict(context or {}).get("exploration_std", self.exploration_std)
        )
        if std < 0:
            raise ValueError("exploration_std cannot be negative")
        action = self._explore(action, std)
        return self._finalize_output(
            {
                "action": action,
                "exploration_std": std,
            }
        )

    def _prepare_chunk_sample(self, chunk: Iterable[MetaTransition]):
        chunk = tuple(chunk)
        if not chunk or not all(isinstance(item, MetaTransition) for item in chunk):
            raise TypeError("training chunks must contain MetaTransition values")

        actions = [np.asarray(item.action.action) for item in chunk[: self.chunk_size]]
        if any(action.ndim != 1 for action in actions):
            raise ValueError("training transitions must contain single-step actions")
        action_dim = actions[0].shape[0]
        if any(action.shape != (action_dim,) for action in actions):
            raise ValueError("training chunk action dimensions must agree")

        valid = len(actions)
        padded = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        padded[:valid] = np.asarray(actions, dtype=np.float32)
        padded = self.meta_policy.action_normalizer.normalize(
            padded,
            datatype="action",
        )
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:valid] = False

        obs = copy.deepcopy(chunk[0].obs)
        normalized_obs = self.meta_policy.state_normalizer.normalize_metaobs(
            obs,
            self.meta_policy.ctrl_space,
        )
        samples = self.meta_policy.normed_mobs_to_samples(normalized_obs)
        if len(samples) != 1:
            raise ValueError("each action chunk must begin with one unbatched MetaObs")
        sample = samples[0]
        sample["action"] = torch.from_numpy(np.asarray(padded)).float()
        sample["is_pad"] = torch.from_numpy(is_pad)
        return sample, valid

    def training_forward(self, batch: Any, *, context=None):
        del context
        chunks = getattr(batch, "chunks", batch)
        if not isinstance(chunks, (list, tuple)) or not chunks:
            raise TypeError("chunk training batch must provide a non-empty chunks sequence")
        prepared = [self._prepare_chunk_sample(chunk) for chunk in chunks]
        samples = [sample for sample, _ in prepared]
        policy_batch = self.meta_policy.meta2obs(samples)
        policy_batch = _move_to_device(policy_batch, _model_device(self.policy))
        result = self.policy(**policy_batch)
        if not isinstance(result, Mapping) or "loss" not in result:
            raise TypeError("native policy forward must return a mapping containing loss")
        output = dict(result)
        output["num_chunks"] = len(samples)
        output["num_actions"] = sum(valid for _, valid in prepared)
        return output

    def reset(self):
        self.meta_policy.reset()

    @staticmethod
    def _checkpoint_root(checkpoint_path):
        root = Path(checkpoint_path)
        if root.name.startswith("checkpoint-"):
            root = root.parent
        return root

    @staticmethod
    def _copy_file(source, destination):
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    def _copy_checkpoint_assets(self, output_dir):
        if self.checkpoint_path is None:
            return
        source_root = self._checkpoint_root(self.checkpoint_path)
        output_root = Path(output_dir)
        metadata_path = source_root / "policy_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"checkpoint metadata was not found: {metadata_path}"
            )
        self._copy_file(metadata_path, output_root / metadata_path.name)

        normalize_path = source_root / "normalize.json"
        if not normalize_path.is_file():
            return
        self._copy_file(normalize_path, output_root / normalize_path.name)
        with normalize_path.open("r", encoding="utf-8") as stream:
            normalize_config = json.load(stream)
        for dataset in normalize_config.get("datasets", ()):
            dataset_id = dataset.get("dataset_id")
            if not dataset_id:
                continue
            ctrl_space = dataset.get("ctrl_space", "ee")
            ctrl_type = dataset.get("ctrl_type", "delta")
            filename = f"{dataset_id}_stats_{ctrl_space}_{ctrl_type}.pkl"
            stats_path = source_root / filename
            if stats_path.is_file():
                self._copy_file(stats_path, output_root / filename)

    def save_pretrained(self, output_dir):
        """Save native model files plus loader metadata and normalizers."""

        result = super().save_pretrained(output_dir)
        self._copy_checkpoint_assets(output_dir)
        return result

    def state_dict(self):
        state = super().state_dict()
        state["rng_state"] = copy.deepcopy(self._rng.bit_generator.state)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "rng_state" not in state:
            raise KeyError("meta-policy adapter state is missing rng_state")
        self._rng.bit_generator.state = state["rng_state"]
