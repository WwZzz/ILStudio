"""Generic action adapters for native, Gaussian and categorical policies."""

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from benchmark.base import MetaAction
from rl.base import PolicyOutput

from .base import ActionAdapter
from ..runtime import (
    explore_action,
    infer_action,
    native_training_forward,
    policy_batch,
    resolve_chunk_size,
)


def _policy_value(result):
    if isinstance(result, Mapping):
        if "action" not in result:
            raise KeyError("policy output must contain 'action'")
        result = result["action"]
    if not torch.is_tensor(result):
        raise TypeError("policy output action must be a tensor")
    if result.ndim == 3:
        if result.shape[1] != 1:
            raise ValueError("standard action adapter requires chunk_size=1")
        result = result[:, 0]
    if result.ndim != 2:
        raise ValueError("policy output must have shape [batch, action_dim]")
    return result


def _forward(policy, batch):
    return policy(**batch) if isinstance(batch, Mapping) else policy(batch)


class StandardActionAdapter(ActionAdapter):
    """Shared policy-forward mechanics for one-step tensor action heads."""

    def __init__(self, meta_policy, *, capabilities):
        super().__init__(meta_policy, capabilities=capabilities)
        self.action_dim = int(self.policy.config.action_dim)
        if resolve_chunk_size(self.policy) != 1:
            raise ValueError("standard action adapters require chunk_size=1")

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def policy_output(self, observations, *, policy=None):
        target = self.policy if policy is None else policy
        return _policy_value(_forward(target, policy_batch(self.meta_policy, observations)))

    def transition_policy_output(self, batch, *, source="obs", policy=None):
        if source not in {"obs", "next_obs"}:
            raise ValueError("source must be obs or next_obs")
        return self.policy_output(
            (getattr(item, source) for item in batch), policy=policy
        )

    def feature_forward(self, observations, *, context=None):
        del context
        return {"policy_output": self.policy_output(observations)}


class GaussianActionAdapter(StandardActionAdapter):
    STATE_VERSION = 1

    def __init__(
        self,
        meta_policy,
        *,
        initial_std=0.2,
        learn_std=True,
        action_low=None,
        action_high=None,
        log_std_min=-5.0,
        log_std_max=2.0,
        gripper_continuous=False,
    ):
        super().__init__(
            meta_policy,
            capabilities={
                "action", "evaluate_actions", "sample_actions",
                "batch_actions", "uniform_actions", "feature_forward",
            },
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if not isinstance(learn_std, bool):
            raise TypeError("learn_std must be bool")
        if not isinstance(initial_std, (int, float)) or float(initial_std) <= 0:
            raise ValueError("initial_std must be positive")
        config = self.policy.config
        low = action_low if action_low is not None else getattr(config, "action_low", None)
        high = action_high if action_high is not None else getattr(config, "action_high", None)
        low = self._bound(low, -1.0, "action_low")
        high = self._bound(high, 1.0, "action_high")
        if np.any(high <= low):
            raise ValueError("each action_high must exceed action_low")
        self.action_low = torch.as_tensor(low, device=self.device)
        self.action_high = torch.as_tensor(high, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dim,), math.log(float(initial_std)), device=self.device
            ),
            requires_grad=learn_std,
        )
        self.learn_std = learn_std
        self.gripper_continuous = bool(gripper_continuous)

    def _bound(self, value, default, name):
        value = [default] * self.action_dim if value is None else value
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 1:
            array = np.repeat(array, self.action_dim)
        if array.size != self.action_dim or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {self.action_dim} finite values")
        return array

    def parameters(self):
        return (self.log_std,) if self.log_std.requires_grad else ()

    def _distribution(self, mean):
        if mean.shape[-1] != self.action_dim:
            raise ValueError("policy output dimension does not match action_dim")
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return Normal(mean, std.expand_as(mean))

    def _squash(self, raw):
        return torch.tanh(raw) * self.action_scale + self.action_bias

    def _unsquash(self, action):
        normalized = (action - self.action_bias) / self.action_scale
        return torch.atanh(normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    def _log_prob(self, distribution, raw):
        squashed = torch.tanh(raw)
        correction = torch.log(self.action_scale * (1.0 - squashed.square()) + 1e-6)
        return (distribution.log_prob(raw) - correction).sum(dim=-1)

    def _sample(self, mean, *, deterministic=False):
        distribution = self._distribution(mean)
        raw = mean if deterministic else distribution.rsample()
        return {
            "action": self._squash(raw),
            "log_prob": self._log_prob(distribution, raw),
            "entropy": distribution.entropy().sum(dim=-1),
            "mean": self._squash(mean),
            "std": distribution.scale,
        }

    def _to_meta_action(self, normalized_action, policy_input):
        action = self.meta_policy.act2meta(
            normalized_action,
            ctrl_space=self.meta_policy.ctrl_space,
            ctrl_type=self.meta_policy.ctrl_type,
        )
        action = self.meta_policy.action_normalizer.denormalize_metaact(action)
        post_process = getattr(self.policy, "post_process_action", None)
        if callable(post_process):
            action.action = post_process(
                action.action,
                policy_input,
                self.meta_policy.action_normalizer,
                self.meta_policy.state_normalizer,
            )
        action.gripper_continuous = self.gripper_continuous
        return action

    def select_action(self, obs, *, deterministic=False, context=None):
        del context
        self.validate_observation(obs)
        inputs = policy_batch(self.meta_policy, (obs,))
        with torch.no_grad():
            sampled = self._sample(_policy_value(_forward(self.policy, inputs)), deterministic=deterministic)
        action = self._to_meta_action(
            sampled["action"][0].detach().cpu().numpy()[None, :], inputs
        )
        return PolicyOutput(
            action=action,
            policy_info={
                "log_prob": float(sampled["log_prob"][0].detach().cpu()),
                "entropy": float(sampled["entropy"][0].detach().cpu()),
                "policy_std": sampled["std"][0].detach().cpu().numpy(),
            },
        )

    def batch_actions(self, batch, *, context=None):
        del context
        values = []
        for item in batch:
            value = np.asarray(item.action.action).reshape(-1)
            if value.size != self.action_dim:
                raise ValueError("stored action dimension does not match action_dim")
            values.append(value)
        normalized = self.meta_policy.action_normalizer.normalize(
            np.asarray(values), datatype="action"
        )
        return torch.as_tensor(normalized, dtype=torch.float32, device=self.device)

    def uniform_actions(self, batch, *, num_samples, context=None):
        del context
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch_size = len(tuple(batch))
        if batch_size <= 0:
            raise ValueError("action sampling requires a non-empty batch")
        shape = (num_samples, batch_size, self.action_dim)
        unit = torch.rand(shape, dtype=torch.float32, device=self.device)
        actions = self.action_low + unit * (self.action_high - self.action_low)
        log_density = -torch.log(self.action_high - self.action_low).sum()
        return {
            "action": actions,
            "log_prob": log_density.expand(num_samples, batch_size),
        }

    def evaluate_actions(self, batch, *, context=None):
        del context
        mean = self.transition_policy_output(batch)
        raw = self._unsquash(self.batch_actions(batch))
        distribution = self._distribution(mean)
        return {
            "log_prob": self._log_prob(distribution, raw),
            "entropy": distribution.entropy().sum(dim=-1),
        }

    def sample_actions(self, batch, *, source="obs", deterministic=False, policy=None, context=None):
        del context
        return self._sample(
            self.transition_policy_output(batch, source=source, policy=policy),
            deterministic=deterministic,
        )

    def clamp_actions(self, actions):
        return torch.maximum(torch.minimum(actions, self.action_high), self.action_low)

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "log_std": self.log_std.detach().cpu(),
            "learn_std": self.learn_std,
        }

    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported Gaussian action adapter state")
        self.log_std.data.copy_(state["log_std"].to(self.log_std.device))


class CategoricalActionAdapter(StandardActionAdapter):
    def __init__(self, meta_policy):
        super().__init__(
            meta_policy,
            capabilities={
                "action", "evaluate_actions", "batch_actions",
                "action_scores", "feature_forward",
            },
        )
        if self.action_dim < 2:
            raise ValueError("categorical action_dim must be at least two")

    def _sample(self, logits, *, deterministic=False, epsilon=0.0):
        distribution = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        elif epsilon > 0:
            greedy = logits.argmax(dim=-1)
            random = torch.randint(self.action_dim, greedy.shape, device=greedy.device)
            action = torch.where(torch.rand_like(greedy.float()) < epsilon, random, greedy)
        else:
            action = distribution.sample()
        return action, distribution

    def select_action(self, obs, *, deterministic=False, context=None):
        self.validate_observation(obs)
        epsilon = float(dict(context or {}).get("discrete_epsilon", 0.0))
        if not 0 <= epsilon <= 1:
            raise ValueError("discrete_epsilon must be in [0, 1]")
        with torch.no_grad():
            action, distribution = self._sample(
                self.policy_output((obs,)), deterministic=deterministic, epsilon=epsilon
            )
        value = action[0]
        return PolicyOutput(
            action=MetaAction(
                ctrl_space=self.meta_policy.ctrl_space,
                ctrl_type=self.meta_policy.ctrl_type,
                action=np.asarray([int(value.cpu())]),
            ),
            policy_info={
                "log_prob": float(distribution.log_prob(action)[0].cpu()),
                "entropy": float(distribution.entropy()[0].cpu()),
            },
        )

    def batch_actions(self, batch, *, context=None):
        del context
        values = [int(np.asarray(item.action.action).reshape(-1)[0]) for item in batch]
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def evaluate_actions(self, batch, *, context=None):
        del context
        logits = self.transition_policy_output(batch)
        distribution = Categorical(logits=logits)
        actions = self.batch_actions(batch)
        return {
            "log_prob": distribution.log_prob(actions),
            "entropy": distribution.entropy(),
            "logits": logits,
        }

    def action_scores(self, batch, *, source="obs", policy=None, context=None):
        del context
        return self.transition_policy_output(batch, source=source, policy=policy)


class NativeActionAdapter(ActionAdapter):
    def __init__(
        self,
        meta_policy,
        *,
        exploration_std=0.0,
        exploration_clip=None,
        chunk_size=None,
        seed=None,
    ):
        super().__init__(meta_policy, capabilities={"action", "training_forward"})
        if not isinstance(exploration_std, (int, float)) or exploration_std < 0:
            raise ValueError("exploration_std must be non-negative")
        self.exploration_std = float(exploration_std)
        self.exploration_clip = None if exploration_clip is None else tuple(exploration_clip)
        if self.exploration_clip is not None and (
            len(self.exploration_clip) != 2 or self.exploration_clip[0] >= self.exploration_clip[1]
        ):
            raise ValueError("exploration_clip must be (low, high)")
        self.chunk_size = resolve_chunk_size(self.policy, chunk_size)
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs, *, deterministic=False, context=None):
        self.validate_observation(obs)
        std = 0.0 if deterministic else float(
            dict(context or {}).get("exploration_std", self.exploration_std)
        )
        action = explore_action(
            self.meta_policy,
            infer_action(self.meta_policy, obs),
            rng=self.rng,
            std=std,
            clip=self.exploration_clip,
        )
        return {"action": action, "exploration_std": std}

    def training_forward(self, batch, *, context=None):
        del context
        return native_training_forward(
            self.meta_policy, batch, chunk_size=self.chunk_size
        )

    def state_dict(self):
        return {"version": self.STATE_VERSION, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.rng.bit_generator.state = state["rng_state"]


__all__ = [
    "CategoricalActionAdapter",
    "GaussianActionAdapter",
    "NativeActionAdapter",
    "StandardActionAdapter",
]
