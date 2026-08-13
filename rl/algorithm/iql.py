"""Implicit Q-Learning for fixed offline datasets."""

from numbers import Real

import torch
import torch.nn.functional as F

from rl.critic import BaseCritic
from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .ddpg import _action_batch
from .utils import (
    bootstrap_discounts,
    decision_rewards,
    detached_metric,
    transitions,
    validate_policy_result,
    vector,
)


def expectile_loss(residual, expectile):
    """Asymmetric squared loss used to fit V below high dataset Q values."""

    weight = torch.where(
        residual >= 0,
        residual.new_tensor(float(expectile)),
        residual.new_tensor(1.0 - float(expectile)),
    )
    return (weight * residual.square()).mean()


def _per_sample_action_mse(actor_actions, replay_actions, action_mask):
    if actor_actions.shape != replay_actions.shape:
        raise ValueError("actor and replay actions must have the same shape")
    squared = (actor_actions - replay_actions).square()
    if action_mask is None:
        return squared.flatten(start_dim=1).mean(dim=1)
    mask = torch.as_tensor(
        action_mask,
        dtype=squared.dtype,
        device=squared.device,
    )
    if mask.shape != squared.shape[:-1]:
        raise ValueError("action_mask must match action tensor prefix")
    denominator = mask.sum(dim=1) * squared.shape[-1]
    if torch.any(denominator <= 0):
        raise ValueError("each action mask must select at least one action")
    return (squared * mask.unsqueeze(-1)).flatten(start_dim=1).sum(dim=1) / denominator


class IQLAlgorithm(BaseRLAlgorithm):
    """Twin-Q, expectile-value and advantage-weighted policy regression."""

    def __init__(
        self,
        *,
        gamma=0.99,
        reward_key="train/total",
        expectile=0.7,
        advantage_temperature=3.0,
        max_advantage_weight=100.0,
        actor_loss="log_prob",
        target_tau=0.005,
        critic=None,
        actor_learning_rate=None,
        critic_learning_rate=None,
        value_learning_rate=None,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(expectile, Real) or not 0.0 < float(expectile) < 1.0:
            raise ValueError("expectile must be in (0, 1)")
        for name, value in (
            ("advantage_temperature", advantage_temperature),
            ("max_advantage_weight", max_advantage_weight),
            ("target_tau", target_tau),
        ):
            if not isinstance(value, Real) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(target_tau) > 1.0:
            raise ValueError("target_tau must not exceed one")
        if actor_loss not in {"log_prob", "mse"}:
            raise ValueError("actor_loss must be log_prob or mse")
        if not isinstance(critic, BaseCritic):
            raise TypeError("IQL requires a configured BaseCritic")
        for name in ("target", "value", "parameter_groups", "soft_update"):
            if not callable(getattr(critic, name, None)):
                raise TypeError(f"IQL critic must provide {name}()")
        for name, value in (
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
            ("value_learning_rate", value_learning_rate),
        ):
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        capabilities = {"action", "batch_actions"}
        capabilities.add("evaluate_actions" if actor_loss == "log_prob" else "sample_actions")
        super().__init__(
            required_capabilities=capabilities,
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.expectile = float(expectile)
        self.advantage_temperature = float(advantage_temperature)
        self.max_advantage_weight = float(max_advantage_weight)
        self.actor_loss = actor_loss
        self.target_tau = float(target_tau)
        self.critic = critic
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.value_learning_rate = value_learning_rate

    @staticmethod
    def _actor_parameters(policy_adapter):
        method = getattr(policy_adapter, "actor_parameters", None)
        if callable(method):
            return tuple(method())
        return tuple(policy_adapter.policy.parameters())

    @staticmethod
    def _twin(value, *, prefix):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("IQL requires twin Q values")
        return vector(value[0], name=f"{prefix}1"), vector(
            value[1], name=f"{prefix}2"
        )

    def parameters(self):
        return tuple(self.critic.parameters())

    def optimizer_parameter_groups(self, policy_adapter):
        critic_groups = dict(self.critic.parameter_groups())
        expected = {"critic1", "critic2", "value"}
        if set(critic_groups) != expected:
            raise ValueError(
                "IQL critic parameter groups must be critic1, critic2, and value"
            )
        groups = {}
        for name, parameters in critic_groups.items():
            learning_rate = (
                self.value_learning_rate if name == "value" else self.critic_learning_rate
            )
            groups[name] = {
                "parameters": parameters,
                "learning_rate": learning_rate,
            }
        groups["actor"] = {
            "parameters": self._actor_parameters(policy_adapter),
            "learning_rate": self.actor_learning_rate,
        }
        return groups

    def _policy_loss(
        self,
        batch,
        *,
        policy_adapter,
        replay_actions,
        action_mask,
        weights,
        context,
    ):
        if self.actor_loss == "log_prob":
            result = validate_policy_result(
                policy_adapter.evaluate_actions(batch, context=context),
                operation="evaluate_actions",
                required=("log_prob",),
            )
            log_prob = vector(result["log_prob"], name="actor_log_prob")
            return -(weights * log_prob).mean()
        result = validate_policy_result(
            policy_adapter.sample_actions(
                batch,
                source="obs",
                deterministic=True,
                context=context,
            ),
            operation="sample_actions",
            required=("action",),
        )
        per_sample = _per_sample_action_mse(
            result["action"], replay_actions, action_mask
        )
        return (weights * per_sample).mean()

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        replay_actions, action_mask = _action_batch(
            policy_adapter.batch_actions(batch, context=context)
        )
        q1, q2 = self._twin(
            self.critic(
                observations,
                replay_actions,
                action_mask=action_mask,
                context=context,
            ),
            prefix="q",
        )
        value = vector(
            self.critic.value(observations, context=context),
            name="value",
        )
        with torch.no_grad():
            next_value = vector(
                self.critic.value(next_observations, context=context),
                name="next_value",
            )
            target_q1, target_q2 = self._twin(
                self.critic.target(
                    observations,
                    replay_actions,
                    action_mask=action_mask,
                    context=context,
                ),
                prefix="target_q",
            )
            target_q = torch.minimum(target_q1, target_q2)
            advantage = target_q - value.detach()
            weights = torch.exp(
                self.advantage_temperature * advantage
            ).clamp(max=self.max_advantage_weight)
        bellman_target = decision_rewards(
            items, self.reward_key, self.gamma, like=q1
        ) + bootstrap_discounts(items, self.gamma, like=q1) * next_value
        q1_loss = F.mse_loss(q1, bellman_target)
        q2_loss = F.mse_loss(q2, bellman_target)
        value_loss = expectile_loss(target_q - value, self.expectile)
        actor_loss = self._policy_loss(
            batch,
            policy_adapter=policy_adapter,
            replay_actions=replay_actions,
            action_mask=action_mask,
            weights=weights,
            context=context,
        )

        def fresh_actor_loss():
            return self._policy_loss(
                batch,
                policy_adapter=policy_adapter,
                replay_actions=replay_actions,
                action_mask=action_mask,
                weights=weights,
                context={**dict(context or {}), "phase": "actor"},
            )

        return AlgorithmOutput(
            loss={
                "critic1": q1_loss,
                "critic2": q2_loss,
                "value": value_loss,
                "actor": fresh_actor_loss,
            },
            metrics={
                "iql/critic1_loss": detached_metric(q1_loss),
                "iql/critic2_loss": detached_metric(q2_loss),
                "iql/value_loss": detached_metric(value_loss),
                "iql/actor_loss": detached_metric(actor_loss),
                "iql/advantage_mean": detached_metric(advantage.mean()),
                "iql/weight_mean": detached_metric(weights.mean()),
                "iql/target_mean": detached_metric(bellman_target.mean()),
            },
            payload={
                "update_order": ("critic1", "critic2", "value", "actor"),
                "post_step": "iql_target",
                "post_step_context": {"tau": self.target_tau},
            },
        )

    def algorithm_post_step(self, operation, *, policy_adapter, context=None):
        if operation != "iql_target":
            return super().algorithm_post_step(
                operation,
                policy_adapter=policy_adapter,
                context=context,
            )
        del policy_adapter
        self.critic.soft_update(dict(context or {}).get("tau", self.target_tau))

    def state_dict(self):
        state = super().state_dict()
        state["critic"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.critic.load_state_dict(state["critic"])


__all__ = ["IQLAlgorithm", "expectile_loss"]
