"""Thin ACT bindings for generic observation and action primitives."""

import copy
from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.base import MetaObs
from rl.base import DecisionTransition, PolicyOutput
from rl.policy_adapter import BasePolicyAdapter, OptimizerTrainerAdapter
from rl.policy_adapter.action import ActionAdapter
from rl.policy_adapter.runtime import (
    explore_action,
    infer_action,
    model_device,
    move_to_device,
    resolve_chunk_size,
)


def _validate_chunk_exploration(*, std, rho, clip):
    if isinstance(std, bool) or not isinstance(std, Real) or float(std) < 0.0:
        raise ValueError("chunk noise std must be non-negative")
    if isinstance(rho, bool) or not isinstance(rho, Real) or not 0.0 <= float(rho) < 1.0:
        raise ValueError("chunk noise rho must be in [0, 1)")
    if clip is not None and (
        isinstance(clip, bool) or not isinstance(clip, Real) or float(clip) <= 0.0
    ):
        raise ValueError("chunk noise clip must be positive or None")
    return float(std), float(rho), None if clip is None else float(clip)


def _chunk_ar1_noise(rng, shape, *, std, rho, clip):
    std, rho, clip = _validate_chunk_exploration(std=std, rho=rho, clip=clip)
    noise = rng.normal(0.0, std, size=shape).astype(np.float32)
    innovation_scale = np.sqrt(1.0 - rho * rho)
    for index in range(1, shape[1]):
        noise[:, index] = rho * noise[:, index - 1] + innovation_scale * noise[:, index]
    if clip is not None:
        np.clip(noise, -clip, clip, out=noise)
    return noise


class ACTActionAdapter(ActionAdapter):
    """Translate ACT chunks and backbone features without owning RL state.

    Critics, target networks, exploration schedules, losses, and optimizer
    ordering are intentionally absent; those belong to algorithm objects.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        meta_policy,
        *,
        action_clip=(-1.0, 1.0),
        exploration_noise_std=0.02,
        exploration_noise_rho=0.9,
        exploration_noise_clip=0.05,
        exploration_std=0.02,
        exploration_clip=(-1.0, 1.0),
        chunk_size=None,
        seed=None,
        feature_spatial_grid=(4, 4),
        feature_timestep_scale=128.0,
    ):
        if chunk_size is None:
            native_chunk = getattr(getattr(meta_policy.policy, "config", None), "chunk_size", None)
            if isinstance(native_chunk, int) and not isinstance(native_chunk, bool):
                chunk_size = min(16, native_chunk)
        super().__init__(
            meta_policy,
            capabilities={
                "action", "sample_actions", "batch_actions",
                "critic_features", "features",
            },
        )
        self.chunk_size = resolve_chunk_size(self.policy, chunk_size)
        self._rng = np.random.default_rng(seed)
        if not isinstance(exploration_std, Real) or float(exploration_std) < 0:
            raise ValueError("exploration_std must be non-negative")
        self.exploration_std = float(exploration_std)
        self.exploration_clip = None if exploration_clip is None else tuple(exploration_clip)
        if self.exploration_clip is not None and (
            len(self.exploration_clip) != 2
            or self.exploration_clip[0] >= self.exploration_clip[1]
        ):
            raise ValueError("exploration_clip must be (low, high)")
        if action_clip is not None:
            action_clip = tuple(float(value) for value in action_clip)
            if len(action_clip) != 2 or action_clip[0] >= action_clip[1]:
                raise ValueError("action_clip must be (low, high) or None")
        if isinstance(feature_spatial_grid, int):
            feature_spatial_grid = (feature_spatial_grid, feature_spatial_grid)
        feature_spatial_grid = tuple(feature_spatial_grid)
        if len(feature_spatial_grid) != 2 or any(int(value) <= 0 for value in feature_spatial_grid):
            raise ValueError("feature_spatial_grid must contain two positive integers")
        if float(feature_timestep_scale) <= 0.0:
            raise ValueError("feature_timestep_scale must be positive")
        self.action_clip = action_clip
        (
            self.exploration_noise_std,
            self.exploration_noise_rho,
            self.exploration_noise_clip,
        ) = _validate_chunk_exploration(
            std=exploration_noise_std,
            rho=exploration_noise_rho,
            clip=exploration_noise_clip,
        )
        self.feature_spatial_grid = tuple(int(value) for value in feature_spatial_grid)
        self.feature_timestep_scale = float(feature_timestep_scale)

    @staticmethod
    def _observations(value):
        if isinstance(value, MetaObs):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = tuple(value)
            if result and all(isinstance(item, MetaObs) for item in result):
                return result
        raise TypeError("ACT adapter requires MetaObs values")

    def _policy_batch(self, observations):
        samples = []
        for obs in self._observations(observations):
            normalized = self.meta_policy.state_normalizer.normalize_metaobs(
                copy.deepcopy(obs), self.meta_policy.ctrl_space
            )
            items = self.meta_policy.normed_mobs_to_samples(normalized)
            if len(items) != 1:
                raise ValueError("ACT adapter requires one sample per observation")
            samples.append(items[0])
        batch = self.meta_policy.meta2obs(samples)
        return move_to_device(batch, model_device(self.policy))

    def _actor_chunks(self, observations, *, policy=None, grad=True):
        policy = self.policy if policy is None else policy
        observations = self._observations(observations)
        batch = self._policy_batch(observations)
        qpos = batch["qpos"]
        latent_dim = getattr(getattr(policy, "model", None), "latent_dim", None)
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
            raise TypeError("ACT model must expose integer latent_dim")
        latent = qpos.new_zeros((qpos.shape[0], latent_dim))
        scope = torch.enable_grad() if grad else torch.no_grad()
        with scope:
            chunks = policy(
                qpos=qpos,
                image=batch["image"],
                latent_sample=latent,
            )
        if not torch.is_tensor(chunks) or chunks.ndim != 3:
            raise ValueError("ACT policy must return [batch, chunk, action]")
        if chunks.shape[1] < self.chunk_size:
            raise ValueError("ACT policy returned fewer actions than chunk_size")
        return chunks[:, : self.chunk_size]

    def select_action(self, obs, *, deterministic=False, context=None):
        self.validate_observation(obs)
        std = 0.0 if deterministic else float(
            dict(context or {}).get("exploration_std", self.exploration_std)
        )
        action = explore_action(
            self.meta_policy,
            infer_action(self.meta_policy, obs),
            rng=self._rng,
            std=std,
            clip=self.exploration_clip,
        )
        normalized = self.meta_policy.action_normalizer.normalize(
            np.asarray(action.action).copy(), datatype="action"
        )
        return PolicyOutput(
            action=action,
            policy_info={
                "normalized_action": np.asarray(normalized, dtype=np.float32),
                "exploration_std": std,
            },
        )

    def _explore_chunks(self, chunks, *, deterministic, context):
        configured = dict(context or {})
        std, rho, clip = _validate_chunk_exploration(
            std=configured.get("exploration_noise_std", self.exploration_noise_std),
            rho=configured.get("exploration_noise_rho", self.exploration_noise_rho),
            clip=configured.get("exploration_noise_clip", self.exploration_noise_clip),
        )
        if deterministic or std == 0.0:
            return chunks, torch.zeros_like(chunks), 0.0
        noise = torch.as_tensor(
            _chunk_ar1_noise(
                self._rng, tuple(chunks.shape), std=std, rho=rho, clip=clip
            ),
            dtype=chunks.dtype,
            device=chunks.device,
        )
        explored = chunks + noise
        bounds = getattr(self, "action_clip", getattr(self, "target_action_clip", None))
        if bounds is not None:
            explored = explored.clamp(*bounds)
        return explored, explored - chunks, std

    def sample_actions(
        self,
        batch,
        *,
        source="obs",
        deterministic=False,
        policy=None,
        context=None,
    ):
        if source not in {"obs", "next_obs"}:
            raise ValueError("source must be obs or next_obs")
        observations = tuple(getattr(item, source) for item in batch)
        actions = self._actor_chunks(
            observations,
            policy=policy,
            grad=policy is None,
        )
        if not deterministic:
            actions, _, _ = self._explore_chunks(
                actions, deterministic=False, context=context
            )
        return {"action": self.clamp_actions(actions)}

    def batch_actions(self, batch, *, context=None):
        del context
        actions = []
        masks = []
        for item in batch:
            if isinstance(item, DecisionTransition):
                value = item.decision.extras.get("normalized_action")
                mask = np.zeros(item.decision.chunk_size, dtype=bool)
                for step in item.steps:
                    mask[step.action_offset] = True
                raw_action = item.decision.action.action
            else:
                value = item.policy_info.get("normalized_action")
                raw_action = item.action.action
            if value is None:
                value = self.meta_policy.action_normalizer.normalize(
                    np.asarray(raw_action).copy(), datatype="action"
                )
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2:
                raise ValueError("ACT replay action must have shape [chunk, action]")
            if not isinstance(item, DecisionTransition):
                mask = np.ones(value.shape[0], dtype=bool)
            actions.append(value)
            masks.append(mask)
        device = model_device(self.policy)
        return {
            "action": torch.as_tensor(np.stack(actions), device=device),
            "action_mask": torch.as_tensor(np.stack(masks), device=device),
        }

    def clamp_actions(self, actions):
        return actions if self.action_clip is None else actions.clamp(*self.action_clip)

    def critic_features(self, obs, *, context=None):
        policy = dict(context or {}).get("feature_policy", self.policy)
        observations = self._observations(obs)
        batch = self._policy_batch(observations)
        image = batch["image"]
        state = batch["qpos"]
        if image.ndim != 5 or state.ndim != 2:
            raise ValueError("ACT features need [B,N,C,H,W] images and [B,S] state")
        model = getattr(policy, "model", None)
        backbones = getattr(model, "backbones", None)
        normalize = getattr(policy, "normalize", None)
        if not backbones or not callable(normalize):
            raise TypeError("ACT policy must expose backbones and normalize")
        with torch.no_grad():
            image = normalize(image)
            camera_features = []
            for camera_index in range(image.shape[1]):
                features, _ = backbones[0](image[:, camera_index])
                camera_features.append(
                    F.adaptive_avg_pool2d(
                        features[-1], self.feature_spatial_grid
                    ).flatten(start_dim=1)
                )
            visual = torch.cat(camera_features, dim=-1)
        timestep = state.new_tensor(
            [max(int(item.timestep), 0) for item in observations]
        ).unsqueeze(-1)
        state = torch.cat(
            [state, timestep / self.feature_timestep_scale], dim=-1
        )
        return {"visual": visual.detach(), "state": state.detach()}

    def features(self, observations, *, context=None):
        return self.critic_features(observations, context=context)

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "base": super().state_dict(),
            "feature_spatial_grid": self.feature_spatial_grid,
            "feature_timestep_scale": self.feature_timestep_scale,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported ACT adapter state version")
        super().load_state_dict(state["base"])
        self._rng.bit_generator.state = state["rng_state"]


def build_rl_adapter(*, model_components, required_capabilities=(), **kwargs):
    policy = model_components["model"]
    meta_policy = model_components["meta_policy"]
    if getattr(meta_policy, "policy", None) is not policy:
        raise ValueError("ACT model and MetaPolicy must share the policy")
    action_adapter = kwargs.pop("action_adapter", None)
    if action_adapter is None:
        action_adapter = ACTActionAdapter(meta_policy, **kwargs)
    elif kwargs:
        raise TypeError("ACT action_adapter config cannot be combined with legacy arguments")
    return BasePolicyAdapter(
        meta_policy,
        action_adapter=action_adapter,
        required_capabilities=required_capabilities,
        checkpoint_path=model_components.get("checkpoint_path"),
    )


def build_trainer_adapter(
    *, policy_components, optimizer=None, scheduler=None, step_fn=None, **kwargs
):
    del policy_components
    if kwargs:
        raise TypeError("unsupported ACT trainer arguments: " + ", ".join(sorted(kwargs)))
    return OptimizerTrainerAdapter(
        optimizer=optimizer,
        scheduler=scheduler,
        step_fn=step_fn,
    )


__all__ = [
    "ACTActionAdapter",
    "build_rl_adapter",
    "build_trainer_adapter",
]
