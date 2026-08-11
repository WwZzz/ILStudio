"""Clipped Proximal Policy Optimization for discrete or continuous policies."""

from numbers import Real

import torch
import torch.nn.functional as F

from rl.policy_adapter import MetaPolicyAdapter
from rl.critic import BaseCritic

from .base import AlgorithmOutput, BaseRLAlgorithm
from .objective import BasePolicyObjectiveBuilder
from .on_policy import FullRolloutUpdates
from .utils import (
    bootstrap_masks,
    detached_metric,
    normalized,
    rewards,
    transition_values,
    transitions,
    validate_policy_result,
    vector,
)


class PPOAlgorithm(FullRolloutUpdates, BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        reward_key: str = "train/total",
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        normalize_advantage: bool = True,
        objective_builder=None,
        critic=None,
    ):
        for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
            if not isinstance(value, Real) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isinstance(clip_ratio, Real) or float(clip_ratio) <= 0.0:
            raise ValueError("clip_ratio must be positive")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(value_coef, Real) or float(value_coef) < 0.0:
            raise ValueError("value_coef must be non-negative")
        if not isinstance(entropy_coef, Real) or float(entropy_coef) < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if not isinstance(normalize_advantage, bool):
            raise TypeError("normalize_advantage must be bool")
        if objective_builder is not None and not isinstance(
            objective_builder, BasePolicyObjectiveBuilder
        ):
            raise TypeError(
                "objective_builder must inherit BasePolicyObjectiveBuilder or be None"
            )
        if not isinstance(critic, BaseCritic):
            raise TypeError("PPO requires a configured critic inheriting BaseCritic")
        super().__init__(
            required_capabilities=(
                "action",
                "recompute_traces" if objective_builder is not None else "evaluate_actions",
            ),
            required_buffer_type="rollout",
        )
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.reward_key = reward_key
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.normalize_advantage = normalize_advantage
        self.objective_builder = objective_builder
        self.critic = critic
        self._cached_rollout_key = None
        self._cached_rollout_values = None
        self._cached_decision_rollout_key = None
        self._cached_decision_rollout_values = None

    def _old_rollout_values(self, items, result, *, like):
        old_log_prob = transition_values(
            items,
            lambda item: item.policy_info["log_prob"],
            like=like,
            name="policy_info log_prob",
        )
        if all("value" in item.policy_info for item in items):
            old_value = transition_values(
                items,
                lambda item: item.policy_info["value"],
                like=like,
                name="policy_info value",
            )
        else:
            old_value = vector(result["value"], name="value").detach()

        forwarded_next = result.get("next_value")
        if forwarded_next is not None:
            forwarded_next = vector(forwarded_next, name="next_value").detach()
            if len(forwarded_next) != len(items):
                raise ValueError("PPO next_value must align with transitions")
        next_values = []
        for index, item in enumerate(items):
            if "next_value" in item.policy_info:
                next_values.append(item.policy_info["next_value"])
            elif item.terminated:
                next_values.append(0.0)
            elif not item.episode_done and index + 1 < len(items):
                next_values.append(old_value[index + 1])
            elif forwarded_next is not None:
                next_values.append(forwarded_next[index])
            else:
                raise KeyError(
                    "PPO needs next_value for a truncated or incomplete rollout tail"
                )
        old_next_value = like.new_tensor(
            [float(value.detach()) if hasattr(value, "detach") else float(value) for value in next_values]
        )
        return old_log_prob, old_value, old_next_value

    def _advantages(self, items, reward, old_value, old_next_value):
        delta = reward + self.gamma * bootstrap_masks(items, like=old_value) * old_next_value - old_value
        advantage = torch.empty_like(delta)
        running = torch.zeros((), dtype=delta.dtype, device=delta.device)
        for index in range(len(items) - 1, -1, -1):
            continuation = 0.0 if items[index].episode_done else 1.0
            running = delta[index] + self.gamma * self.gae_lambda * continuation * running
            advantage[index] = running
        return advantage

    def _decision_values(self, groups, result, *, like):
        if all(group.decision.value is not None for group in groups):
            old_value = like.new_tensor(
                [float(group.decision.value) for group in groups]
            )
        else:
            old_value = like.detach()
        forwarded_next = result.get("next_value")
        if forwarded_next is not None:
            forwarded_next = vector(forwarded_next, name="next_value").detach()
            if len(forwarded_next) != len(groups):
                raise ValueError("PPO next_value must align with policy decisions")
        next_values = []
        for index, group in enumerate(groups):
            if "next_value" in group.decision.extras:
                next_values.append(group.decision.extras["next_value"])
            elif group.terminated:
                next_values.append(0.0)
            elif not group.episode_done and index + 1 < len(groups):
                next_values.append(old_value[index + 1])
            elif forwarded_next is not None:
                next_values.append(forwarded_next[index])
            else:
                raise KeyError(
                    "decision PPO needs next_value for a truncated or incomplete tail"
                )
        old_next_value = like.new_tensor(
            [
                float(value.detach()) if hasattr(value, "detach") else float(value)
                for value in next_values
            ]
        )
        return old_value, old_next_value

    def _decision_advantages(self, groups, reward, old_value, old_next_value):
        bootstrap = old_value.new_tensor(
            [0.0 if group.terminated else 1.0 for group in groups]
        )
        delta = reward + self.gamma * bootstrap * old_next_value - old_value
        advantage = torch.empty_like(delta)
        running = torch.zeros((), dtype=delta.dtype, device=delta.device)
        for index in range(len(groups) - 1, -1, -1):
            continuation = 0.0 if groups[index].episode_done else 1.0
            running = (
                delta[index]
                + self.gamma * self.gae_lambda * continuation * running
            )
            advantage[index] = running
        return advantage

    def _compute_decision_update(self, batch, *, policy_adapter, context):
        rollout = getattr(batch, "rollout", None)
        if rollout is None:
            raise ValueError(
                "decision-level PPO requires a rollout-aware RolloutBuffer batch"
            )
        groups = rollout.decision_transitions()
        required = ("traces",) if self.critic is not None else ("traces", "value")
        result = dict(validate_policy_result(
            policy_adapter.recompute_traces(batch, context=context),
            operation="recompute_traces",
            required=required,
        ))
        if self.critic is not None:
            value = self.critic(
                tuple(group.decision.obs for group in groups),
                context=context,
            )
            with torch.no_grad():
                next_value = self.critic(
                    tuple(group.steps[-1].transition.next_obs for group in groups),
                    context=context,
                )
            result["value"] = value
            result["next_value"] = next_value.detach()
        value = vector(result["value"], name="value")
        if len(value) != len(groups):
            raise ValueError("PPO values must align with policy decisions")
        rollout_key = (
            tuple(id(step.transition) for step in rollout.steps),
            tuple(group.decision.decision_id for group in groups),
            tuple(
                group.decision.extras.get("policy_version") for group in groups
            ),
        )
        if rollout_key != self._cached_decision_rollout_key:
            old_value, old_next_value = self._decision_values(
                groups, result, like=value
            )
            reward = value.new_tensor(
                [group.reward_sum(self.reward_key) for group in groups]
            )
            advantage = self._decision_advantages(
                groups, reward, old_value, old_next_value
            )
            self._cached_decision_rollout_key = rollout_key
            self._cached_decision_rollout_values = (
                advantage.detach(),
                (advantage + old_value).detach(),
            )
        advantage, returns = self._cached_decision_rollout_values
        policy_advantage = (
            normalized(advantage) if self.normalize_advantage else advantage
        )
        objective = self.objective_builder.build(
            rollout,
            result["traces"],
            policy_advantage,
            values=value,
            returns=returns,
        )
        mask = objective.mask
        if mask is None:
            mask = torch.ones_like(objective.new_logprobs, dtype=torch.bool)
        ratio = objective.ratio
        unclipped = ratio * objective.advantages.detach()
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio,
        ) * objective.advantages.detach()
        policy_loss = -torch.minimum(unclipped, clipped).masked_select(mask).mean()
        value_loss = F.mse_loss(value, returns.detach())
        entropy = result.get("entropy")
        if entropy is None:
            entropy_mean = value.new_zeros(())
        else:
            entropy = torch.as_tensor(entropy, device=value.device)
            entropy_mean = entropy.mean()
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy_mean
        )
        delta_log_prob = (
            objective.old_logprobs - objective.new_logprobs
        ).masked_select(mask)
        clipped_mask = ((ratio - 1.0).abs() > self.clip_ratio).masked_select(mask)
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "ppo/loss": detached_metric(loss),
                "ppo/policy_loss": detached_metric(policy_loss),
                "ppo/value_loss": detached_metric(value_loss),
                "ppo/approx_kl": detached_metric(delta_log_prob.mean()),
                "ppo/clip_fraction": detached_metric(clipped_mask.float().mean()),
                "ppo/num_decisions": len(groups),
                "ppo/num_objectives": int(mask.sum()),
            },
        )

    def state_dict(self):
        state = super().state_dict()
        state["critic"] = (
            None if self.critic is None else self.critic.state_dict()
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        critic_state = state.get("critic")
        if critic_state is not None:
            if self.critic is None:
                raise ValueError("PPO state contains a critic but algorithm has none")
            self.critic.load_state_dict(critic_state)

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        if self.objective_builder is not None:
            return self._compute_decision_update(
                batch,
                policy_adapter=policy_adapter,
                context=context,
            )
        items = transitions(batch)
        result = dict(validate_policy_result(
            policy_adapter.evaluate_actions(batch, context=context),
            operation="evaluate_actions",
            required=("log_prob",),
        ))
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        result["value"] = self.critic(observations, context=context)
        with torch.no_grad():
            result["next_value"] = self.critic(
                next_observations, context=context
            ).detach()
        log_prob = vector(result["log_prob"], name="log_prob")
        value = vector(result["value"], name="value")
        if len(log_prob) != len(items) or len(value) != len(items):
            raise ValueError("PPO outputs must align with transitions")
        rollout_key = (
            tuple(id(item) for item in items),
            tuple(int(index) for index in batch.indices),
            tuple(item.policy_info.get("policy_version") for item in items),
        )
        if rollout_key != self._cached_rollout_key:
            old_log_prob, old_value, old_next_value = self._old_rollout_values(
                items, result, like=value
            )
            reward = rewards(items, self.reward_key, like=value)
            advantage = self._advantages(items, reward, old_value, old_next_value)
            self._cached_rollout_key = rollout_key
            self._cached_rollout_values = (
                old_log_prob.detach(),
                advantage.detach(),
                (advantage + old_value).detach(),
            )
        old_log_prob, advantage, returns = self._cached_rollout_values
        policy_advantage = normalized(advantage) if self.normalize_advantage else advantage
        ratio = torch.exp(log_prob - old_log_prob)
        unclipped = ratio * policy_advantage.detach()
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio,
        ) * policy_advantage.detach()
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.mse_loss(value, returns.detach())
        entropy = vector(result.get("entropy", value.new_zeros(len(items))), name="entropy")
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "ppo/loss": detached_metric(loss),
                "ppo/policy_loss": detached_metric(policy_loss),
                "ppo/value_loss": detached_metric(value_loss),
                "ppo/approx_kl": detached_metric((old_log_prob - log_prob).mean()),
                "ppo/clip_fraction": detached_metric(
                    ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                ),
            },
        )
