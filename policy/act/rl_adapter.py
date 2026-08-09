"""ACT-specific reinforcement-learning hooks."""

import copy
import inspect
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.base import MetaAction, MetaObs
from rl.base import DecisionTransition
from rl.critic import PolicyFeatureChunkQCritic
from rl.policy_adapter import BasicTrainerAdapter
from rl.policy_adapter.meta_policy import (
    MetaPolicyAdapter,
    _model_device,
    _move_to_device,
)


_STATE_FILENAME = "rl_adapter.pt"


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)


def _validate_chunk_exploration(*, std, rho, clip):
    if isinstance(std, bool) or not isinstance(std, Real) or float(std) < 0.0:
        raise ValueError("chunk noise std must be non-negative")
    if (
        isinstance(rho, bool)
        or not isinstance(rho, Real)
        or not 0.0 <= float(rho) < 1.0
    ):
        raise ValueError("chunk noise rho must be in [0, 1)")
    if clip is not None and (
        isinstance(clip, bool)
        or not isinstance(clip, Real)
        or float(clip) <= 0.0
    ):
        raise ValueError("chunk noise clip must be positive or None")
    return float(std), float(rho), None if clip is None else float(clip)


def _chunk_ar1_noise(rng, shape, *, std, rho, clip):
    """Sample stationary AR(1) noise along the chunk time dimension."""

    std, rho, clip = _validate_chunk_exploration(
        std=std,
        rho=rho,
        clip=clip,
    )
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in shape
    ):
        raise ValueError("chunk exploration shape must be [batch, chunk, action]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("chunk exploration rng must be numpy Generator")
    noise = rng.normal(0.0, std, size=shape).astype(np.float32)
    innovation_scale = np.sqrt(1.0 - rho * rho)
    for index in range(1, shape[1]):
        noise[:, index] = (
            rho * noise[:, index - 1]
            + innovation_scale * noise[:, index]
        )
    if clip is not None:
        np.clip(noise, -clip, clip, out=noise)
    return noise


def _chunk_ar1_noise_like(reference, *, std, rho, clip):
    """Torch equivalent used by TD3 target-policy smoothing."""

    if not torch.is_tensor(reference) or reference.ndim != 3:
        raise ValueError("chunk noise reference must be a rank-3 tensor")
    std, rho, clip = _validate_chunk_exploration(
        std=std,
        rho=rho,
        clip=clip,
    )
    if std == 0.0:
        return torch.zeros_like(reference)
    noise = torch.randn_like(reference).mul_(std)
    innovation_scale = np.sqrt(1.0 - rho * rho)
    for index in range(1, reference.shape[1]):
        noise[:, index] = (
            rho * noise[:, index - 1]
            + innovation_scale * noise[:, index]
        )
    if clip is not None:
        noise.clamp_(-clip, clip)
    return noise


class ACTRLPolicyAdapter(MetaPolicyAdapter):
    """ACT prior sampling plus detached visual features for value/Q learning."""

    STATE_VERSION = 3

    def __init__(
        self,
        meta_policy,
        *,
        latent_std=1.0,
        algorithm=None,
        critic_hidden_dim=256,
        critic_activation="relu",
        actor_learning_rate=None,
        critic_learning_rate=None,
        target_action_clip=(-1.0, 1.0),
        exploration_noise_std=0.02,
        exploration_noise_rho=0.9,
        exploration_noise_clip=0.05,
        target_policy_noise_std=0.02,
        target_policy_noise_rho=0.9,
        target_policy_noise_clip=0.05,
        critic_spatial_grid=(4, 4),
        critic_timestep_scale=128.0,
        **kwargs,
    ):
        if isinstance(latent_std, bool) or not isinstance(latent_std, Real):
            raise TypeError("latent_std must be a real number")
        if float(latent_std) <= 0.0:
            raise ValueError("latent_std must be positive")
        if algorithm not in {None, "ddpg", "td3"}:
            raise ValueError("ACT RL adapter algorithm must be ddpg, td3, or None")
        if target_action_clip is not None:
            target_action_clip = tuple(target_action_clip)
            if (
                len(target_action_clip) != 2
                or not all(isinstance(value, Real) for value in target_action_clip)
                or float(target_action_clip[0]) >= float(target_action_clip[1])
            ):
                raise ValueError("target_action_clip must be (low, high) or None")
            target_action_clip = tuple(float(value) for value in target_action_clip)
        if isinstance(critic_spatial_grid, int) and not isinstance(
            critic_spatial_grid, bool
        ):
            critic_spatial_grid = (critic_spatial_grid, critic_spatial_grid)
        else:
            critic_spatial_grid = tuple(critic_spatial_grid)
        if (
            len(critic_spatial_grid) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in critic_spatial_grid
            )
        ):
            raise ValueError("critic_spatial_grid must contain two positive integers")
        if (
            isinstance(critic_timestep_scale, bool)
            or not isinstance(critic_timestep_scale, Real)
            or float(critic_timestep_scale) <= 0.0
        ):
            raise ValueError("critic_timestep_scale must be positive")
        for name, value in (
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive or None")
        if algorithm in {"ddpg", "td3"} and kwargs.get("chunk_size") is None:
            policy_config = getattr(meta_policy.policy, "config", None)
            native_chunk_size = getattr(policy_config, "chunk_size", None)
            if not isinstance(native_chunk_size, int) or isinstance(
                native_chunk_size, bool
            ):
                raise TypeError("ACT policy config must expose integer chunk_size")
            # Keep the checkpoint's native prediction horizon while replanning
            # more frequently for closed-loop RL control.
            kwargs["chunk_size"] = min(16, native_chunk_size)
        super().__init__(meta_policy, exploration_std=0.0, **kwargs)
        self.latent_std = float(latent_std)
        self.algorithm = algorithm
        self.actor_learning_rate = (
            None if actor_learning_rate is None else float(actor_learning_rate)
        )
        self.critic_learning_rate = (
            None if critic_learning_rate is None else float(critic_learning_rate)
        )
        self.target_action_clip = target_action_clip
        self.critic_spatial_grid = critic_spatial_grid
        self.critic_timestep_scale = float(critic_timestep_scale)
        (
            self.exploration_noise_std,
            self.exploration_noise_rho,
            self.exploration_noise_clip,
        ) = _validate_chunk_exploration(
            std=exploration_noise_std,
            rho=exploration_noise_rho,
            clip=exploration_noise_clip,
        )
        (
            self.target_policy_noise_std,
            self.target_policy_noise_rho,
            self.target_policy_noise_clip,
        ) = _validate_chunk_exploration(
            std=target_policy_noise_std,
            rho=target_policy_noise_rho,
            clip=target_policy_noise_clip,
        )
        capabilities = set(self.capabilities)
        capabilities.update(
            {
                "critic_features",
                "latent_sampling",
            }
        )
        self.target_policy = None
        self.q1 = None
        self.q2 = None
        self.target_q1 = None
        self.target_q2 = None
        self._targets_initialized = False
        if algorithm in {"ddpg", "td3"}:
            capabilities.update({algorithm, "target_update"})
            self.target_policy = _freeze(copy.deepcopy(self.policy))
            self.q1 = PolicyFeatureChunkQCritic(
                self,
                hidden_dim=critic_hidden_dim,
                activation=critic_activation,
                detach_features=True,
                use_language=False,
            ).to(_model_device(self.policy))
            self.target_q1 = PolicyFeatureChunkQCritic(
                self,
                hidden_dim=critic_hidden_dim,
                activation=critic_activation,
                detach_features=True,
                use_language=False,
            ).to(_model_device(self.policy))
            self.target_q1.eval()
            if algorithm == "td3":
                self.q2 = PolicyFeatureChunkQCritic(
                    self,
                    hidden_dim=critic_hidden_dim,
                    activation=critic_activation,
                    detach_features=True,
                    use_language=False,
                ).to(_model_device(self.policy))
                self.target_q2 = PolicyFeatureChunkQCritic(
                    self,
                    hidden_dim=critic_hidden_dim,
                    activation=critic_activation,
                    detach_features=True,
                    use_language=False,
                ).to(_model_device(self.policy))
                self.target_q2.eval()
        self._capabilities = frozenset(capabilities)

    @staticmethod
    def _observations(obs):
        if isinstance(obs, MetaObs):
            return (obs,)
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            observations = tuple(obs)
            if observations and all(
                isinstance(item, MetaObs) for item in observations
            ):
                return observations
        raise TypeError(
            "ACT critic features require MetaObs or a non-empty MetaObs sequence"
        )

    def _critic_batch(self, observations):
        samples = []
        for obs in observations:
            normalized = self.meta_policy.state_normalizer.normalize_metaobs(
                copy.deepcopy(obs),
                self.meta_policy.ctrl_space,
            )
            items = self.meta_policy.normed_mobs_to_samples(normalized)
            if len(items) != 1:
                raise ValueError(
                    "ACT critic features require one sample per observation"
                )
            samples.append(items[0])
        batch = self.meta_policy.meta2obs(samples)
        return _move_to_device(batch, _model_device(self.policy))

    def _latent_sample(self, batch, *, deterministic, context):
        act_model = getattr(self.policy, "model", None)
        latent_dim = getattr(act_model, "latent_dim", None)
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
            raise TypeError("ACT policy model must expose integer latent_dim")
        std = dict(context or {}).get("latent_std", self.latent_std)
        if isinstance(std, bool) or not isinstance(std, Real) or float(std) <= 0.0:
            raise ValueError("latent_std must be positive")
        qpos = batch["qpos"]
        shape = (qpos.shape[0], latent_dim)
        if deterministic:
            return qpos.new_zeros(shape), 0.0
        value = self._rng.normal(0.0, float(std), size=shape).astype(np.float32)
        return torch.as_tensor(value, device=qpos.device, dtype=qpos.dtype), float(std)

    def _policy_chunks(self, observations, *, deterministic, context):
        observations = self._observations(observations)
        batch = self._critic_batch(observations)
        if "image" not in batch or "qpos" not in batch:
            raise KeyError("ACT inference batch requires image and qpos")
        latent, used_std = self._latent_sample(
            batch, deterministic=deterministic, context=context
        )
        was_training = bool(getattr(self.policy, "training", False))
        self.policy.eval()
        try:
            with torch.no_grad():
                chunk = self.policy(
                    qpos=batch["qpos"],
                    image=batch["image"],
                    latent_sample=latent,
                )
        finally:
            self.policy.train(was_training)
        if (
            not torch.is_tensor(chunk)
            or chunk.ndim != 3
            or chunk.shape[0] != len(observations)
        ):
            raise ValueError("ACT policy must return [batch, chunk, action] tensor")
        if chunk.shape[1] < self.chunk_size:
            raise ValueError("ACT policy returned a shorter action chunk than configured")
        return chunk[:, : self.chunk_size], batch, latent, used_std

    def _policy_chunk(self, obs, *, deterministic, context):
        chunks, batch, latent, used_std = self._policy_chunks(
            (obs,), deterministic=deterministic, context=context
        )
        return chunks[0], batch, latent, used_std

    def _explore_chunks(self, chunks, *, deterministic, context):
        configured = dict(context or {})
        std, rho, clip = _validate_chunk_exploration(
            std=configured.get(
                "exploration_noise_std", self.exploration_noise_std
            ),
            rho=configured.get(
                "exploration_noise_rho", self.exploration_noise_rho
            ),
            clip=configured.get(
                "exploration_noise_clip", self.exploration_noise_clip
            ),
        )
        if (
            deterministic
            or std == 0.0
            or self.algorithm not in {"ddpg", "td3"}
        ):
            return chunks, torch.zeros_like(chunks), 0.0
        sampled = _chunk_ar1_noise(
            self._rng,
            tuple(chunks.shape),
            std=std,
            rho=rho,
            clip=clip,
        )
        noise = torch.as_tensor(
            sampled,
            dtype=chunks.dtype,
            device=chunks.device,
        )
        explored = chunks + noise
        if self.target_action_clip is not None:
            explored = explored.clamp(*self.target_action_clip)
        return explored, explored - chunks, std

    def _smooth_target_chunks(self, chunks, *, context):
        configured = dict(context or {})
        std, rho, clip = _validate_chunk_exploration(
            std=configured.get(
                "target_policy_noise_std", self.target_policy_noise_std
            ),
            rho=configured.get(
                "target_policy_noise_rho", self.target_policy_noise_rho
            ),
            clip=configured.get(
                "target_policy_noise_clip", self.target_policy_noise_clip
            ),
        )
        noise = _chunk_ar1_noise_like(
            chunks,
            std=std,
            rho=rho,
            clip=clip,
        )
        smoothed = chunks + noise
        if self.target_action_clip is not None:
            smoothed = smoothed.clamp(*self.target_action_clip)
        return smoothed, smoothed - chunks

    @staticmethod
    def _slice_policy_batch(value, index, batch_size):
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[index : index + 1]
            return value
        if isinstance(value, Mapping):
            return {
                key: ACTRLPolicyAdapter._slice_policy_batch(
                    item, index, batch_size
                )
                for key, item in value.items()
            }
        if isinstance(value, tuple) and len(value) == batch_size:
            return (value[index],)
        if isinstance(value, list) and len(value) == batch_size:
            return [value[index]]
        return value

    def _to_meta_action(self, normalized_action, policy_batch):
        action = self.meta_policy.act2meta(
            normalized_action.detach().unsqueeze(0),
            ctrl_space=self.meta_policy.ctrl_space,
            ctrl_type=self.meta_policy.ctrl_type,
        )
        action = self.meta_policy.action_normalizer.denormalize_metaact(action)
        post_process = getattr(self.policy, "post_process_action", None)
        if callable(post_process):
            action.action = post_process(
                action.action,
                policy_batch,
                self.meta_policy.action_normalizer,
                self.meta_policy.state_normalizer,
            )
        value = action.action
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError("ACT post-processed action must have shape [chunk, action]")
        return MetaAction(
            action=value.astype(np.float32, copy=False),
            ctrl_space=action.ctrl_space,
            ctrl_type=action.ctrl_type,
            gripper_continuous=action.gripper_continuous,
        )

    def select_action(self, obs, *, deterministic=False, context=None):
        return self.select_actions(
            (obs,),
            deterministic=deterministic,
            context=context,
        )[0]

    def select_actions(self, observations, *, deterministic=False, context=None):
        observations = self._observations(observations)
        chunks, batch, latent, used_std = self._policy_chunks(
            observations,
            deterministic=deterministic,
            context=context,
        )
        actor_chunks = chunks
        chunks, exploration_noise, used_noise_std = self._explore_chunks(
            actor_chunks,
            deterministic=deterministic,
            context=context,
        )
        outputs = []
        batch_size = len(observations)
        for index, chunk in enumerate(chunks):
            policy_batch = self._slice_policy_batch(batch, index, batch_size)
            outputs.append(
                self._finalize_output(
                    {
                        "action": self._to_meta_action(chunk, policy_batch),
                        "latent_sample": latent[index : index + 1]
                        .detach()
                        .cpu()
                        .numpy(),
                        "latent_std": used_std,
                        "exploration": (
                            "deterministic"
                            if deterministic
                            else (
                                "act_prior_chunk_ar1" if used_noise_std else "act_prior"
                            )
                        ),
                        "exploration_noise": exploration_noise[index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        "exploration_noise_std": used_noise_std,
                        "actor_normalized_action": actor_chunks[index]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        "normalized_action": chunk.detach().float().cpu().numpy(),
                    }
                )
            )
        return tuple(outputs)

    def critic_features(self, obs, *, context=None):
        feature_policy = dict(context or {}).get("feature_policy", self.policy)
        observations = self._observations(obs)
        batch = self._critic_batch(observations)
        try:
            image = batch["image"]
            state = batch["qpos"]
        except KeyError as exc:
            raise KeyError(
                "ACT critic feature batch requires image and qpos"
            ) from exc
        if image.ndim != 5 or state.ndim != 2:
            raise ValueError(
                "ACT critic inputs must be [B,N,C,H,W] images and [B,S] state"
            )

        act_model = getattr(feature_policy, "model", None)
        backbones = getattr(act_model, "backbones", None)
        if not backbones:
            raise TypeError("ACT policy does not expose a visual backbone")
        backbone = backbones[0]
        normalize = getattr(feature_policy, "normalize", None)
        if not callable(normalize):
            raise TypeError("ACT policy does not expose image normalization")

        was_training = bool(getattr(feature_policy, "training", False))
        feature_policy.eval()
        try:
            with torch.no_grad():
                image = normalize(image)
                camera_features = []
                for camera_index in range(image.shape[1]):
                    features, _ = backbone(image[:, camera_index])
                    if not features or features[-1].ndim != 4:
                        raise ValueError(
                            "ACT backbone must return spatial feature maps"
                        )
                    camera_features.append(
                        F.adaptive_avg_pool2d(
                            features[-1], self.critic_spatial_grid
                        ).flatten(start_dim=1)
                    )
                visual = torch.cat(camera_features, dim=-1)
        finally:
            feature_policy.train(was_training)
        timestep = state.new_tensor(
            [max(int(observation.timestep), 0) for observation in observations]
        ).unsqueeze(-1)
        state = torch.cat(
            [state, timestep / self.critic_timestep_scale], dim=-1
        )
        return {
            "visual": visual.detach(),
            "state": state.detach(),
        }

    def parameters(self):
        if self.q1 is None:
            return ()
        modules = (self.q1,) if self.q2 is None else (self.q1, self.q2)
        return tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    def actor_parameters(self):
        return tuple(
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        )

    def critic_parameters(self):
        return self.parameters()

    def set_training(self, training: bool) -> None:
        super().set_training(training)
        if self.q1 is not None:
            self.q1.train(training)
        if self.q2 is not None:
            self.q2.train(training)
        if self.target_policy is not None:
            self.target_policy.eval()
        if self.target_q1 is not None:
            self.target_q1.eval()
        if self.target_q2 is not None:
            self.target_q2.eval()

    def _actor_chunks(self, observations, *, policy, grad):
        batch = self._critic_batch(observations)
        qpos = batch["qpos"]
        act_model = getattr(policy, "model", None)
        latent_dim = getattr(act_model, "latent_dim", None)
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
            raise TypeError("ACT policy model must expose integer latent_dim")
        latent = qpos.new_zeros((qpos.shape[0], latent_dim))
        was_training = bool(getattr(policy, "training", False))
        policy.eval()
        try:
            context = torch.enable_grad() if grad else torch.no_grad()
            with context:
                chunk = policy(
                    qpos=qpos,
                    image=batch["image"],
                    latent_sample=latent,
                )
        finally:
            policy.train(was_training)
        if (
            not torch.is_tensor(chunk)
            or chunk.ndim != 3
            or chunk.shape[0] != len(observations)
            or chunk.shape[1] < self.chunk_size
        ):
            raise ValueError("ACT actor must return [batch, chunk, action]")
        return chunk[:, : self.chunk_size]

    def _decision_actions(self, groups, *, key="normalized_action"):
        actions = []
        masks = []
        for group in groups:
            value = group.decision.extras.get(key)
            if value is None:
                raise KeyError(
                    "ACT deterministic actor-critic replay requires decision " + key
                    + " metadata"
                )
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 2 or value.shape[0] != self.chunk_size:
                raise ValueError(
                    "ACT normalized action must have shape [configured chunk, action]"
                )
            mask = np.zeros(self.chunk_size, dtype=bool)
            for step in group.steps:
                mask[step.action_offset] = True
            actions.append(value)
            masks.append(mask)
        device = _model_device(self.policy)
        return (
            torch.as_tensor(np.stack(actions), device=device),
            torch.as_tensor(np.stack(masks), device=device),
        )

    @staticmethod
    def _masked_action_mse(actor_actions, target_actions, action_mask):
        mask = action_mask.to(dtype=actor_actions.dtype).unsqueeze(-1)
        squared_error = (actor_actions - target_actions).square() * mask
        denominator = mask.sum() * actor_actions.shape[-1]
        if denominator.item() <= 0:
            raise ValueError("ACT actor BC mask must execute at least one action")
        return squared_error.sum() / denominator

    @staticmethod
    def _decision_groups(batch):
        groups = tuple(batch)
        if not groups or not all(
            isinstance(group, DecisionTransition) for group in groups
        ):
            raise TypeError(
                "ACT deterministic actor-critic requires DecisionTransition batches"
            )
        return groups

    def _initialize_target_q(self, observations, actions, action_mask):
        if self._targets_initialized:
            return
        with torch.no_grad():
            self.target_q1(
                observations,
                actions,
                action_mask=action_mask,
            )
        self.target_q1.load_state_dict(self.q1.state_dict())
        _freeze(self.target_q1)
        if self.q2 is not None:
            with torch.no_grad():
                self.target_q2(
                    observations,
                    actions,
                    action_mask=action_mask,
                )
            self.target_q2.load_state_dict(self.q2.state_dict())
            _freeze(self.target_q2)
        self._targets_initialized = True

    def algorithm_forward(self, operation, batch, *, context=None):
        if self.algorithm not in {"ddpg", "td3"} or operation != self.algorithm:
            raise ValueError(
                f"ACT adapter configured for {self.algorithm!r}, got {operation!r}"
            )
        groups = self._decision_groups(batch)
        observations = tuple(group.obs for group in groups)
        full_mask = torch.ones(
            (len(groups), self.chunk_size),
            dtype=torch.bool,
            device=_model_device(self.policy),
        )
        phase = None
        # DDPG asks for a fresh actor forward after the critic optimizer step.
        if isinstance(context, Mapping):
            phase = context.get("phase")
        replay_actions, replay_mask = self._decision_actions(groups)
        actor_targets, actor_target_mask = self._decision_actions(
            groups, key="actor_normalized_action"
        )
        if phase == "actor":
            actor_actions = self._actor_chunks(
                observations,
                policy=self.policy,
                grad=True,
            )
            actor_q1 = self.q1(
                observations,
                actor_actions,
                action_mask=full_mask,
            )
            key = "actor_q" if self.algorithm == "ddpg" else "actor_q1"
            actor_bc_loss = self._masked_action_mse(
                actor_actions,
                actor_targets,
                actor_target_mask,
            )
            return {key: actor_q1, "actor_bc_loss": actor_bc_loss}

        critic_q1 = self.q1(
            observations,
            replay_actions,
            action_mask=replay_mask,
        )
        critic_q2 = None
        if self.q2 is not None:
            critic_q2 = self.q2(
                observations,
                replay_actions,
                action_mask=replay_mask,
            )
        next_observations = tuple(group.next_obs for group in groups)
        with torch.no_grad():
            actor_actions = self._actor_chunks(
                observations,
                policy=self.policy,
                grad=False,
            )
            actor_q = self.q1(
                observations,
                actor_actions,
                action_mask=full_mask,
            )
            actor_bc_loss = self._masked_action_mse(
                actor_actions,
                actor_targets,
                actor_target_mask,
            )
            target_actions = self._actor_chunks(
                next_observations,
                policy=self.target_policy,
                grad=False,
            )
            if self.algorithm == "td3":
                target_actions, _ = self._smooth_target_chunks(
                    target_actions, context=context
                )
            self._initialize_target_q(
                next_observations,
                target_actions,
                full_mask,
            )
            target_context = {"feature_policy": self.target_policy}
            target_next_q1 = self.target_q1(
                next_observations,
                target_actions,
                action_mask=full_mask,
                context=target_context,
            )
            target_next_q2 = None
            if self.target_q2 is not None:
                target_next_q2 = self.target_q2(
                    next_observations,
                    target_actions,
                    action_mask=full_mask,
                    context=target_context,
                )
        if self.algorithm == "ddpg":
            return {
                "critic_q": critic_q1,
                "target_next_q": target_next_q1,
                "actor_q": actor_q,
                "actor_bc_loss": actor_bc_loss,
            }
        return {
            "critic_q1": critic_q1,
            "critic_q2": critic_q2,
            "target_next_q1": target_next_q1,
            "target_next_q2": target_next_q2,
            "actor_q1": actor_q,
            "actor_bc_loss": actor_bc_loss,
        }

    def algorithm_post_step(self, operation, *, context=None):
        if operation != f"{self.algorithm}_target":
            raise ValueError(f"unsupported ACT target operation {operation!r}")
        tau = float(dict(context or {}).get("tau", 1.0))
        if not 0.0 < tau <= 1.0:
            raise ValueError("target update tau must be in (0, 1]")
        if not self._targets_initialized:
            raise RuntimeError("ACT target Q must be initialized before target update")
        _soft_update(self.target_policy, self.policy, tau)
        _soft_update(self.target_q1, self.q1, tau)
        if self.q2 is not None:
            _soft_update(self.target_q2, self.q2, tau)

    def state_dict(self):
        result = {
            "version": self.STATE_VERSION,
            "base": super().state_dict(),
            "algorithm": self.algorithm,
            "critic_spatial_grid": self.critic_spatial_grid,
            "critic_timestep_scale": self.critic_timestep_scale,
        }
        if self.algorithm in {"ddpg", "td3"}:
            result.update(
                {
                    "q1": self.q1.state_dict(),
                    "target_q1": self.target_q1.state_dict(),
                    "target_policy": self.target_policy.state_dict(),
                    "targets_initialized": self._targets_initialized,
                }
            )
            if self.q2 is not None:
                result.update(
                    {
                        "q2": self.q2.state_dict(),
                        "target_q2": self.target_q2.state_dict(),
                    }
                )
        return result

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("ACT RL adapter state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            if state.get("version") in {1, 2}:
                raise ValueError(
                    "ACT RL adapter state predates the spatial critic; restart RL "
                    "from the policy checkpoint so the critic can be reinitialized"
                )
            raise ValueError("unsupported ACT RL adapter state version")
        saved_grid = tuple(state.get("critic_spatial_grid", ()))
        saved_scale = state.get("critic_timestep_scale")
        if saved_grid != tuple(self.critic_spatial_grid) or saved_scale != (
            self.critic_timestep_scale
        ):
            raise ValueError(
                "ACT RL critic configuration does not match the saved adapter"
            )
        if state.get("algorithm") != self.algorithm:
            raise ValueError("ACT RL adapter algorithm does not match state")
        super().load_state_dict(state["base"])
        if self.algorithm in {"ddpg", "td3"}:
            self.q1.load_state_dict(state["q1"])
            self.target_q1.load_state_dict(state["target_q1"])
            self.target_policy.load_state_dict(state["target_policy"])
            self._targets_initialized = bool(state["targets_initialized"])
            if self._targets_initialized:
                _freeze(self.target_q1)
            if self.q2 is not None:
                self.q2.load_state_dict(state["q2"])
                self.target_q2.load_state_dict(state["target_q2"])
                if self._targets_initialized:
                    _freeze(self.target_q2)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = super().save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir / _STATE_FILENAME)
        return result


def build_rl_adapter(
    *,
    model_components,
    required_capabilities=(),
    **kwargs,
):
    try:
        policy = model_components["model"]
        meta_policy = model_components["meta_policy"]
    except KeyError as exc:
        raise KeyError(
            "ACT RL adapter requires model and meta_policy components"
        ) from exc
    if getattr(meta_policy, "policy", None) is not policy:
        raise ValueError("ACT model and MetaPolicy must share the policy")
    kwargs.setdefault("checkpoint_path", model_components.get("checkpoint_path"))
    requested = frozenset(required_capabilities)
    probabilistic = requested.intersection({"ppo", "reinforce"})
    if probabilistic:
        raise ValueError(
            "ACT inference is deterministic (z=0) and does not expose an action "
            "likelihood; use a deterministic actor-critic adapter, or explicitly "
            "select the generic gaussian_chunk approximation"
        )
    if not requested or requested == {"action"}:
        passthrough = {
            name: kwargs[name]
            for name in ("checkpoint_path", "chunk_size", "seed")
            if name in kwargs
        }
        return MetaPolicyAdapter(meta_policy, **passthrough)
    kwargs.pop("policy_std", None)
    deterministic = requested.intersection({"ddpg", "td3"})
    if len(deterministic) > 1:
        raise ValueError("ACT RL adapter cannot combine DDPG and TD3")
    if deterministic:
        algorithm = next(iter(deterministic))
        kwargs["algorithm"] = algorithm
    adapter = ACTRLPolicyAdapter(meta_policy, **kwargs)
    adapter.require_capabilities(required_capabilities)
    checkpoint_path = model_components.get("checkpoint_path")
    if checkpoint_path:
        path = MetaPolicyAdapter._checkpoint_root(checkpoint_path) / _STATE_FILENAME
        if path.is_file():
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            adapter.load_state_dict(state)
    return adapter


def _clone_optimizer(template, parameters, *, learning_rate=None):
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("ACT trainer adapter received an empty parameter group")
    optimizer_class = type(template)
    defaults = dict(template.defaults)
    if learning_rate is not None:
        defaults["lr"] = float(learning_rate)
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        optimizer_args = defaults
    else:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        optimizer_args = (
            defaults
            if accepts_kwargs
            else {
                name: value
                for name, value in defaults.items()
                if name in signature.parameters
            }
        )
    return optimizer_class(parameters, **optimizer_args)


def build_trainer_adapter(
    *,
    policy_components,
    policy_adapter=None,
    optimizer=None,
    scheduler=None,
    step_fn=None,
    **kwargs,
):
    del policy_components
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported ACT trainer adapter arguments: {unknown}")
    if not isinstance(policy_adapter, ACTRLPolicyAdapter):
        raise TypeError("ACT trainer adapter requires ACTRLPolicyAdapter")
    if optimizer is None:
        raise ValueError("ACT trainer adapter requires the configured optimizer")
    if step_fn is not None:
        raise ValueError("ACT trainer adapter does not accept a custom step_fn")
    if policy_adapter.algorithm in {"ddpg", "td3"}:
        if scheduler is not None:
            raise ValueError("deterministic actor-critic requires per-optimizer schedulers")
        return BasicTrainerAdapter(
            optimizer={
                "critic": _clone_optimizer(
                    optimizer,
                    policy_adapter.critic_parameters(),
                    learning_rate=policy_adapter.critic_learning_rate,
                ),
                "actor": _clone_optimizer(
                    optimizer,
                    policy_adapter.actor_parameters(),
                    learning_rate=policy_adapter.actor_learning_rate,
                ),
            }
        )
    return BasicTrainerAdapter(optimizer=optimizer, scheduler=scheduler)


RLPolicyAdapter = ACTRLPolicyAdapter

__all__ = [
    "ACTRLPolicyAdapter",
    "RLPolicyAdapter",
    "build_rl_adapter",
    "build_trainer_adapter",
]
