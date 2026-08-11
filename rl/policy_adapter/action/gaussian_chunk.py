"""Gaussian action-chunk adapter for continuous policy-gradient algorithms."""

import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Optional

import numpy as np
import torch

from benchmark.base import MetaAction, MetaObs, MetaPolicy
from rl.base import ACTION_POLICY_INFO_KEY, MetaTransition, PolicyTrace

from .base import ActionAdapter
from ..runtime import (
    model_device,
    move_to_device,
    native_training_forward,
    resolve_chunk_size,
)


class GaussianChunkActionAdapter(ActionAdapter):
    """Turn a differentiable action-chunk model into a Gaussian RL policy.

    The existing policy predicts the mean of a diagonal Gaussian in normalized
    action space. Its state-independent per-action log standard deviations are
    learnable by default. Environment actions still pass through the
    checkpoint's normalizer and optional post-processing. The sampled normalized
    action is retained so REINFORCE/PPO can recompute its exact log probability.

    Stochastic base policies such as diffusion policy are replayed with a stored
    per-decision torch seed. This adapter does not model diffusion denoising-step
    likelihoods; it defines an explicit outer Gaussian over the final chunk.
    """

    def __init__(
        self,
        meta_policy: MetaPolicy,
        *,
        policy_std: float,
        learn_fixed_std: bool = True,
        action_dim: Optional[int] = None,
        inference_seed: int = 0,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        policy_std = self._validate_std(policy_std)
        if not isinstance(learn_fixed_std, bool):
            raise TypeError("learn_fixed_std must be bool")
        if not isinstance(inference_seed, Integral) or isinstance(
            inference_seed, bool
        ):
            raise TypeError("inference_seed must be an integer")
        super().__init__(
            meta_policy,
            capabilities={
                "action", "training_forward", "evaluate_actions", "recompute_traces"
            },
        )
        self.chunk_size = resolve_chunk_size(self.policy, chunk_size)
        self._rng = np.random.default_rng(seed)
        self.action_dim = self._resolve_action_dim(action_dim)
        device = model_device(self.policy)
        self.log_std = torch.nn.Parameter(
            torch.full(
                (self.action_dim,),
                math.log(policy_std),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=learn_fixed_std,
        )
        self.learn_fixed_std = learn_fixed_std
        self.inference_seed = int(inference_seed)
        self._decision_index = 0

    @property
    def policy_std(self) -> np.ndarray:
        """Return the current per-action standard deviations."""

        return self.log_std.detach().exp().cpu().numpy().copy()

    def parameters(self):
        """Expose the shared log standard deviation to the composed optimizer."""

        return (self.log_std,) if self.log_std.requires_grad else ()

    def _std_tensor(self, like: torch.Tensor, *, collected_std=None):
        if self.learn_fixed_std:
            return self.log_std.exp().to(dtype=like.dtype, device=like.device)
        if collected_std is None:
            return self.log_std.detach().exp().to(dtype=like.dtype, device=like.device)
        value = self._validate_collected_std(collected_std)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def _distribution(self, mean: torch.Tensor, *, collected_std=None):
        return torch.distributions.Normal(
            mean,
            self._std_tensor(mean, collected_std=collected_std),
        )

    def _resolve_action_dim(self, action_dim) -> int:
        candidates = [action_dim]
        config = getattr(self.policy, "config", None)
        for owner in (self.policy, config):
            if owner is not None:
                candidates.extend(
                    getattr(owner, name, None) for name in ("action_dim", "act_dim")
                )
        action_head = getattr(getattr(self.policy, "model", None), "action_head", None)
        candidates.append(getattr(action_head, "out_features", None))
        for value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        raise ValueError(
            "action_dim is absent from adapter arguments and policy configuration"
        )

    def _policy_batch(self, obs: MetaObs):
        normalized_obs = self.meta_policy.state_normalizer.normalize_metaobs(
            copy.deepcopy(obs),
            self.meta_policy.ctrl_space,
        )
        samples = self.meta_policy.normed_mobs_to_samples(normalized_obs)
        if len(samples) != 1:
            raise ValueError(
                "GaussianChunkActionAdapter supports one synchronous environment"
            )
        policy_batch = self.meta_policy.meta2obs(samples)
        if not isinstance(policy_batch, Mapping):
            raise TypeError("MetaPolicy.meta2obs must return a mapping")
        return move_to_device(policy_batch, model_device(self.policy))

    def _seeded_forward(self, policy_batch, seed: int):
        device = model_device(self.policy) or torch.device("cpu")
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
        was_training = bool(getattr(self.policy, "training", False))
        eval_method = getattr(self.policy, "eval", None)
        train_method = getattr(self.policy, "train", None)
        if callable(eval_method):
            eval_method()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(seed)
                result = self.policy(**policy_batch)
        finally:
            if callable(train_method):
                train_method(was_training)
        return result

    def _chunk_mean(self, obs: MetaObs, seed: int):
        result = self._seeded_forward(self._policy_batch(obs), seed)
        if isinstance(result, Mapping):
            for name in ("action", "actions", "action_chunk"):
                if name in result:
                    result = result[name]
                    break
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                "Gaussian chunk policy forward must return an action tensor"
            )
        if result.ndim == 3:
            if result.shape[0] != 1:
                raise ValueError("Gaussian chunk policy requires batch size one")
            result = result[0]
        elif result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2:
            raise ValueError("policy action tensor must have shape [1, T, A] or [T, A]")
        if result.shape[0] < self.chunk_size:
            raise ValueError(
                f"policy returned {result.shape[0]} actions, expected {self.chunk_size}"
            )
        return result[: self.chunk_size].float()

    def _to_meta_action(self, normalized_action, policy_batch):
        batched = normalized_action.detach().unsqueeze(0)
        action = self.meta_policy.act2meta(
            batched,
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
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError("post-processed action batch must contain one item")
            value = value[0]
        elif value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2:
            raise ValueError("post-processed action must have shape [T, A]")
        return MetaAction(
            action=value.astype(np.float32, copy=False),
            ctrl_space=action.ctrl_space,
            ctrl_type=action.ctrl_type,
            gripper_continuous=action.gripper_continuous,
        )

    @staticmethod
    def _validate_std(value: Any) -> float:
        if not isinstance(value, Real):
            raise TypeError("policy_std must be a real number")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("policy_std must be finite and positive")
        return value

    def _validate_collected_std(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = np.full((self.action_dim,), float(array), dtype=np.float32)
        if array.shape != (self.action_dim,):
            raise ValueError(
                f"policy_std must have shape ({self.action_dim},) in policy metadata"
            )
        if not np.isfinite(array).all() or not np.all(array > 0.0):
            raise ValueError("policy_std must be finite and positive")
        return array

    def select_action(self, obs: MetaObs, *, deterministic=False, context=None):
        self.validate_observation(obs)
        context = dict(context or {})
        if self.learn_fixed_std and "policy_std" in context:
            raise ValueError(
                "runtime policy_std overrides require learn_fixed_std=False"
            )
        std = self._validate_collected_std(
            context.get("policy_std", self.policy_std)
        )
        decision_id = self._decision_index
        action_seed = self.inference_seed + decision_id
        policy_batch = self._policy_batch(obs)
        with torch.no_grad():
            mean = self._seeded_forward(policy_batch, action_seed)
            if isinstance(mean, Mapping):
                for name in ("action", "actions", "action_chunk"):
                    if name in mean:
                        mean = mean[name]
                        break
            if not isinstance(mean, torch.Tensor):
                raise TypeError(
                    "Gaussian chunk policy forward must return an action tensor"
                )
            if mean.ndim == 3:
                if mean.shape[0] != 1:
                    raise ValueError("Gaussian chunk policy requires batch size one")
                mean = mean[0]
            elif mean.ndim == 1:
                mean = mean.unsqueeze(0)
            if mean.ndim != 2 or mean.shape[0] < self.chunk_size:
                raise ValueError("policy action tensor has an invalid chunk shape")
            mean = mean[: self.chunk_size].float()

            mean_array = mean.cpu().numpy()
            if deterministic:
                sampled_array = mean_array.copy()
            else:
                noise = self._rng.normal(0.0, 1.0, size=mean_array.shape) * std
                sampled_array = mean_array + noise.astype(mean_array.dtype)
            sampled = torch.as_tensor(
                sampled_array,
                dtype=mean.dtype,
                device=mean.device,
            )
            distribution = self._distribution(mean, collected_std=std)
            log_prob = distribution.log_prob(sampled).sum(dim=-1)

        action = self._to_meta_action(sampled, policy_batch)
        self._decision_index += 1
        return {
                "action": action,
                "decision_id": decision_id,
                "decision_obs": copy.deepcopy(obs),
                "action_seed": action_seed,
                "policy_std": std.copy(),
                ACTION_POLICY_INFO_KEY: {
                    "sampled_action": sampled_array.copy(),
                    "log_prob": log_prob.cpu().numpy(),
                    "value": np.zeros(self.chunk_size, dtype=np.float32),
                    "next_value": np.zeros(self.chunk_size, dtype=np.float32),
                },
            }

    @staticmethod
    def _batch_transitions(batch):
        items = tuple(getattr(batch, "transitions", batch))
        if not items or not all(isinstance(item, MetaTransition) for item in items):
            raise TypeError("algorithm batch must contain MetaTransition values")
        return items

    def recompute_traces(self, batch, *, context=None):
        del context
        rollout = getattr(batch, "rollout", None)
        if rollout is None:
            raise ValueError("trace recomputation requires a rollout-aware batch")
        traces = {}
        entropies = []
        values = []
        for group in rollout.decision_transitions():
            decision = group.decision
            info = decision.extras
            seed = info.get("action_seed")
            if not isinstance(seed, Integral) or isinstance(seed, bool):
                raise TypeError("decision action_seed must be an integer")
            mean = self._chunk_mean(decision.obs, int(seed))[: decision.chunk_size]
            action_info = info.get(ACTION_POLICY_INFO_KEY)
            if not isinstance(action_info, Mapping):
                raise TypeError("decision is missing per-action policy metadata")
            sampled_action = torch.as_tensor(
                np.asarray(action_info["sampled_action"]),
                dtype=mean.dtype,
                device=mean.device,
            )[: decision.chunk_size]
            if sampled_action.shape != mean.shape:
                raise ValueError("sampled action chunk does not match policy mean")
            std = self._validate_collected_std(info.get("policy_std"))
            distribution = self._distribution(mean, collected_std=std)
            log_prob = distribution.log_prob(sampled_action).sum(dim=-1)
            traces[decision.decision_id] = PolicyTrace(
                kind="chunk",
                old_logprobs=log_prob,
                valid_mask=torch.ones_like(log_prob, dtype=torch.bool),
                axis_names=("action",),
            )
            entropies.append(distribution.entropy().sum(dim=-1))
            values.append(log_prob.new_zeros(()))
        value = torch.stack(values)
        return {
            "traces": traces,
            "value": value,
            "next_value": torch.zeros_like(value),
            "entropy": torch.cat(entropies),
        }

    def evaluate_actions(self, batch, *, context=None):
        del context
        items = self._batch_transitions(batch)
        groups = {}
        for position, transition in enumerate(items):
            info = transition.policy_info
            decision_id = info.get("decision_id")
            if not isinstance(decision_id, Integral) or isinstance(decision_id, bool):
                raise TypeError("policy_info decision_id must be an integer")
            groups.setdefault(int(decision_id), []).append((position, transition))

        log_probs = [None] * len(items)
        entropies = [None] * len(items)
        for group in groups.values():
            first_info = group[0][1].policy_info
            obs = first_info.get("decision_obs")
            seed = first_info.get("action_seed")
            if not isinstance(obs, MetaObs):
                raise TypeError("policy_info decision_obs must be MetaObs")
            if not isinstance(seed, Integral) or isinstance(seed, bool):
                raise TypeError("policy_info action_seed must be an integer")
            mean = self._chunk_mean(obs, int(seed))
            for position, transition in group:
                info = transition.policy_info
                if info.get("action_seed") != seed:
                    raise ValueError("one policy decision has inconsistent action seeds")
                chunk_index = info.get("chunk_index")
                if not isinstance(chunk_index, Integral) or isinstance(chunk_index, bool):
                    raise TypeError("policy_info chunk_index must be an integer")
                chunk_index = int(chunk_index)
                if not 0 <= chunk_index < mean.shape[0]:
                    raise IndexError("policy_info chunk_index is outside the action chunk")
                sampled_action = torch.as_tensor(
                    np.asarray(info["sampled_action"]),
                    dtype=mean.dtype,
                    device=mean.device,
                )
                if sampled_action.shape != mean[chunk_index].shape:
                    raise ValueError("sampled_action shape does not match policy action")
                std = self._validate_collected_std(info.get("policy_std"))
                distribution = self._distribution(
                    mean[chunk_index],
                    collected_std=std,
                )
                log_probs[position] = distribution.log_prob(sampled_action).sum()
                entropies[position] = distribution.entropy().sum()

        log_prob = torch.stack(log_probs)
        entropy = torch.stack(entropies)
        value = log_prob.new_zeros(log_prob.shape)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "next_value": value.clone(),
        }

    def state_dict(self):
        state = super().state_dict()
        state["decision_index"] = self._decision_index
        state["log_std"] = self.log_std.detach().cpu().tolist()
        state["learn_fixed_std"] = self.learn_fixed_std
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        decision_index = state.get("decision_index")
        if not isinstance(decision_index, int) or decision_index < 0:
            raise ValueError("decision_index must be a non-negative integer")
        self._decision_index = decision_index
        if "log_std" in state:
            log_std = np.asarray(state["log_std"], dtype=np.float32)
            if log_std.ndim == 0:
                log_std = np.full((self.action_dim,), float(log_std), dtype=np.float32)
            if log_std.shape != (self.action_dim,) or not np.isfinite(log_std).all():
                raise ValueError(
                    f"log_std must contain {self.action_dim} finite values"
                )
            with torch.no_grad():
                self.log_std.copy_(self.log_std.new_tensor(log_std))
        saved_learn_fixed_std = state.get("learn_fixed_std")
        if saved_learn_fixed_std is not None and not isinstance(
            saved_learn_fixed_std, bool
        ):
            raise TypeError("learn_fixed_std state must be bool")

    def training_forward(self, batch, *, context=None):
        del context
        return native_training_forward(
            self.meta_policy,
            batch,
            chunk_size=self.chunk_size,
        )


__all__ = ["GaussianChunkActionAdapter"]
