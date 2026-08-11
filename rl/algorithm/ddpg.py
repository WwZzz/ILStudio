"""Deep Deterministic Policy Gradient with algorithm-owned targets."""

import copy
from collections.abc import Mapping
from numbers import Real

import torch
import torch.nn.functional as F

from rl.critic import BaseCritic
from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .utils import (
    bootstrap_discounts,
    decision_rewards,
    detached_metric,
    transitions,
    validate_policy_result,
    vector,
)


def _action_batch(value):
    if isinstance(value, Mapping):
        if "action" not in value:
            raise KeyError("batch_actions mapping must contain action")
        return value["action"], value.get("action_mask")
    return value, None


def _masked_action_mse(actor_actions, target_actions, action_mask=None):
    if action_mask is None:
        return F.mse_loss(actor_actions, target_actions)
    mask = torch.as_tensor(
        action_mask, dtype=actor_actions.dtype, device=actor_actions.device
    )
    if mask.shape != actor_actions.shape[:-1]:
        raise ValueError("action_mask must match action tensor prefix")
    expanded = mask.unsqueeze(-1)
    denominator = expanded.sum() * actor_actions.shape[-1]
    if float(denominator.detach()) <= 0.0:
        raise ValueError("action_mask must select at least one action")
    return ((actor_actions - target_actions).square() * expanded).sum() / denominator


class DDPGAlgorithm(BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        target_tau: float = 0.005,
        actor_bc_coef: float = 0.0,
        critic=None,
        actor_learning_rate=None,
        critic_learning_rate=None,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(target_tau, Real) or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if not isinstance(actor_bc_coef, Real) or float(actor_bc_coef) < 0.0:
            raise ValueError("actor_bc_coef must be non-negative")
        if not isinstance(critic, BaseCritic):
            raise TypeError("DDPG requires a configured BaseCritic")
        for name in ("target", "parameter_groups", "soft_update"):
            if not callable(getattr(critic, name, None)):
                raise TypeError(f"DDPG critic must provide {name}()")
        super().__init__(
            required_capabilities=("action", "batch_actions", "sample_actions"),
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.target_tau = float(target_tau)
        self.actor_bc_coef = float(actor_bc_coef)
        self.critic = critic
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.target_policy = None
        self._pending_target_state = None

    @staticmethod
    def _actor_parameters(policy_adapter):
        method = getattr(policy_adapter, "actor_parameters", None)
        if callable(method):
            return tuple(method())
        parameters = list(policy_adapter.policy.parameters())
        parameters.extend(policy_adapter.parameters())
        return tuple(dict.fromkeys(parameters))

    def _ensure_target_policy(self, policy_adapter):
        if self.target_policy is None:
            self.target_policy = copy.deepcopy(policy_adapter.policy)
            self.target_policy.eval()
            self.target_policy.requires_grad_(False)
            if self._pending_target_state is not None:
                self.target_policy.load_state_dict(self._pending_target_state)
                self._pending_target_state = None
        return self.target_policy

    def parameters(self):
        return tuple(self.critic.parameters())

    def optimizer_parameter_groups(self, policy_adapter):
        return {
            "critic": {
                "parameters": tuple(self.critic.parameters()),
                "learning_rate": self.critic_learning_rate,
            },
            "actor": {
                "parameters": self._actor_parameters(policy_adapter),
                "learning_rate": self.actor_learning_rate,
            },
        }

    def _actor_loss(self, batch, items, policy_adapter, context):
        observations = tuple(item.obs for item in items)
        replay_actions, action_mask = _action_batch(
            policy_adapter.batch_actions(batch, context=context)
        )
        actor = validate_policy_result(
            policy_adapter.sample_actions(
                batch, source="obs", deterministic=True, context=context
            ),
            operation="sample_actions",
            required=("action",),
        )
        actor_q = vector(
            self.critic(
                observations,
                actor["action"],
                action_mask=action_mask,
                context=context,
            ),
            name="actor_q",
        )
        bc_loss = _masked_action_mse(actor["action"], replay_actions, action_mask)
        return -actor_q.mean() + self.actor_bc_coef * bc_loss, actor_q, bc_loss

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        replay_actions, action_mask = _action_batch(
            policy_adapter.batch_actions(batch, context=context)
        )
        critic_q = vector(
            self.critic(
                observations,
                replay_actions,
                action_mask=action_mask,
                context=context,
            ),
            name="critic_q",
        )
        target_policy = self._ensure_target_policy(policy_adapter)
        with torch.no_grad():
            target_actor = validate_policy_result(
                policy_adapter.sample_actions(
                    batch,
                    source="next_obs",
                    deterministic=True,
                    policy=target_policy,
                    context=context,
                ),
                operation="sample_actions",
                required=("action",),
            )
            target_next_q = vector(
                self.critic.target(
                    next_observations,
                    target_actor["action"],
                    context={**dict(context or {}), "feature_policy": target_policy},
                ),
                name="target_next_q",
            )
        target = decision_rewards(
            items, self.reward_key, self.gamma, like=critic_q
        ) + bootstrap_discounts(items, self.gamma, like=critic_q) * target_next_q
        critic_loss = F.mse_loss(critic_q, target.detach())
        actor_loss, actor_q, actor_bc_loss = self._actor_loss(
            batch, items, policy_adapter, context
        )

        def fresh_actor_loss():
            return self._actor_loss(batch, items, policy_adapter, context)[0]

        return AlgorithmOutput(
            loss={"critic": critic_loss, "actor": fresh_actor_loss},
            metrics={
                "ddpg/critic_loss": detached_metric(critic_loss),
                "ddpg/actor_loss": detached_metric(actor_loss),
                "ddpg/actor_bc_loss": detached_metric(actor_bc_loss),
                "ddpg/q_mean": detached_metric(critic_q.mean()),
                "ddpg/actor_q_mean": detached_metric(actor_q.mean()),
                "ddpg/target_mean": detached_metric(target.mean()),
            },
            payload={
                "update_order": ("critic", "actor"),
                "post_step": "target_networks",
                "post_step_context": {"tau": self.target_tau},
            },
        )

    def algorithm_post_step(self, operation, *, policy_adapter, context=None):
        if operation != "target_networks":
            return super().algorithm_post_step(
                operation, policy_adapter=policy_adapter, context=context
            )
        tau = float(dict(context or {}).get("tau", self.target_tau))
        self.critic.soft_update(tau)
        target_policy = self._ensure_target_policy(policy_adapter)
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target_policy.parameters(), policy_adapter.policy.parameters()
            ):
                target_parameter.lerp_(source_parameter, tau)

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "critic": self.critic.state_dict(),
                "target_policy": (
                    None if self.target_policy is None else self.target_policy.state_dict()
                ),
            }
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.critic.load_state_dict(state["critic"])
        self._pending_target_state = state.get("target_policy")
