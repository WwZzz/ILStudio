"""Gaussian action-chunk adapter for continuous policy-gradient algorithms."""

import copy
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, Optional

import numpy as np
import torch

from benchmark.base import MetaAction, MetaObs, MetaPolicy
from rl.base import ACTION_POLICY_INFO_KEY, MetaTransition, PolicyTrace

from .meta_policy import MetaPolicyAdapter, _model_device, _move_to_device


class GaussianChunkPolicyAdapter(MetaPolicyAdapter):
    """Turn a differentiable action-chunk model into a Gaussian RL policy.

    The existing policy predicts the mean of a fixed-standard-deviation Gaussian
    in normalized action space. Environment actions still pass through the
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
        inference_seed: int = 0,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        checkpoint_path=None,
    ) -> None:
        if not isinstance(policy_std, Real) or float(policy_std) <= 0.0:
            raise ValueError("policy_std must be positive")
        if not isinstance(inference_seed, Integral) or isinstance(
            inference_seed, bool
        ):
            raise TypeError("inference_seed must be an integer")
        super().__init__(
            meta_policy,
            exploration_std=0.0,
            chunk_size=chunk_size,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        self._capabilities = frozenset(
            {"action", "chunk_training", "reinforce", "ppo"}
        )
        self.policy_std = float(policy_std)
        self.inference_seed = int(inference_seed)
        self._decision_index = 0

    def _policy_batch(self, obs: MetaObs):
        normalized_obs = self.meta_policy.state_normalizer.normalize_metaobs(
            copy.deepcopy(obs),
            self.meta_policy.ctrl_space,
        )
        samples = self.meta_policy.normed_mobs_to_samples(normalized_obs)
        if len(samples) != 1:
            raise ValueError(
                "GaussianChunkPolicyAdapter supports one synchronous environment"
            )
        policy_batch = self.meta_policy.meta2obs(samples)
        if not isinstance(policy_batch, Mapping):
            raise TypeError("MetaPolicy.meta2obs must return a mapping")
        return _move_to_device(policy_batch, _model_device(self.policy))

    def _seeded_forward(self, policy_batch, seed: int):
        device = _model_device(self.policy) or torch.device("cpu")
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
        if not isinstance(value, Real) or float(value) <= 0.0:
            raise ValueError("policy_std must be positive")
        return float(value)

    def select_action(self, obs: MetaObs, *, deterministic=False, context=None):
        self._validate_obs(obs)
        std = self._validate_std(
            dict(context or {}).get("policy_std", self.policy_std)
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
                noise = self._rng.normal(0.0, std, size=mean_array.shape)
                sampled_array = mean_array + noise.astype(mean_array.dtype)
            sampled = torch.as_tensor(
                sampled_array,
                dtype=mean.dtype,
                device=mean.device,
            )
            distribution = torch.distributions.Normal(mean, std)
            log_prob = distribution.log_prob(sampled).sum(dim=-1)

        action = self._to_meta_action(sampled, policy_batch)
        self._decision_index += 1
        return self._finalize_output(
            {
                "action": action,
                "decision_id": decision_id,
                "decision_obs": copy.deepcopy(obs),
                "action_seed": action_seed,
                "policy_std": std,
                ACTION_POLICY_INFO_KEY: {
                    "sampled_action": sampled_array.copy(),
                    "log_prob": log_prob.cpu().numpy(),
                    "value": np.zeros(self.chunk_size, dtype=np.float32),
                    "next_value": np.zeros(self.chunk_size, dtype=np.float32),
                },
            }
        )

    @staticmethod
    def _batch_transitions(batch):
        items = tuple(getattr(batch, "transitions", batch))
        if not items or not all(isinstance(item, MetaTransition) for item in items):
            raise TypeError("algorithm batch must contain MetaTransition values")
        return items

    def algorithm_forward(self, operation: str, batch, *, context=None):
        del context
        if operation == "ppo_trace":
            rollout = getattr(batch, "rollout", None)
            if rollout is None:
                raise ValueError("ppo_trace requires a rollout-aware batch")
            traces = {}
            entropies = []
            values = []
            for group in rollout.decision_transitions():
                decision = group.decision
                info = decision.extras
                seed = info.get("action_seed")
                if not isinstance(seed, Integral) or isinstance(seed, bool):
                    raise TypeError("decision action_seed must be an integer")
                mean = self._chunk_mean(decision.obs, int(seed))
                mean = mean[: decision.chunk_size]
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
                std = self._validate_std(info.get("policy_std"))
                distribution = torch.distributions.Normal(mean, std)
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
        if operation not in {"reinforce", "ppo"}:
            raise ValueError(f"unsupported Gaussian chunk operation {operation!r}")
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
                std = self._validate_std(info.get("policy_std"))
                distribution = torch.distributions.Normal(mean[chunk_index], std)
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
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        decision_index = state.get("decision_index")
        if not isinstance(decision_index, int) or decision_index < 0:
            raise ValueError("decision_index must be a non-negative integer")
        self._decision_index = decision_index
