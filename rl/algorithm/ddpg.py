"""Deep Deterministic Policy Gradient for continuous actions."""

from numbers import Real

import torch.nn.functional as F

from rl.policy_adapter import BasePolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .utils import (
    bootstrap_discounts,
    decision_rewards,
    detached_metric,
    forward_result,
    transitions,
    vector,
)


class DDPGAlgorithm(BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        target_tau: float = 0.005,
        actor_bc_coef: float = 0.0,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(target_tau, Real) or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if not isinstance(actor_bc_coef, Real) or float(actor_bc_coef) < 0.0:
            raise ValueError("actor_bc_coef must be non-negative")
        super().__init__(
            required_capabilities=("action", "ddpg", "target_update"),
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.target_tau = float(target_tau)
        self.actor_bc_coef = float(actor_bc_coef)

    def compute_update(self, batch, *, policy_adapter: BasePolicyAdapter, context=None):
        items = transitions(batch)
        result = forward_result(
            policy_adapter,
            "ddpg",
            batch,
            context=context,
            required=("critic_q", "target_next_q", "actor_q"),
        )
        critic_q = vector(result["critic_q"], name="critic_q")
        target_next_q = vector(result["target_next_q"], name="target_next_q")
        actor_q = vector(result["actor_q"], name="actor_q")
        if any(len(value) != len(items) for value in (critic_q, target_next_q, actor_q)):
            raise ValueError("DDPG outputs must align with transitions")
        target = decision_rewards(
            items, self.reward_key, self.gamma, like=critic_q
        ) + bootstrap_discounts(
            items, self.gamma, like=critic_q
        ) * target_next_q.detach()
        critic_loss = F.mse_loss(critic_q, target)
        actor_bc_loss = result.get("actor_bc_loss")
        if actor_bc_loss is None:
            actor_bc_loss = actor_q.new_zeros(())
        elif not hasattr(actor_bc_loss, "numel") or actor_bc_loss.numel() != 1:
            raise ValueError("DDPG actor_bc_loss must be a scalar tensor")
        actor_loss = -actor_q.mean() + self.actor_bc_coef * actor_bc_loss

        def fresh_actor_loss():
            actor_result = forward_result(
                policy_adapter,
                "ddpg",
                batch,
                context={**dict(context or {}), "phase": "actor"},
                required=("actor_q",),
            )
            fresh_actor_q = vector(actor_result["actor_q"], name="actor_q")
            if len(fresh_actor_q) != len(items):
                raise ValueError("DDPG actor_q must align with transitions")
            fresh_bc_loss = actor_result.get("actor_bc_loss")
            if fresh_bc_loss is None:
                fresh_bc_loss = fresh_actor_q.new_zeros(())
            elif not hasattr(fresh_bc_loss, "numel") or fresh_bc_loss.numel() != 1:
                raise ValueError("DDPG actor_bc_loss must be a scalar tensor")
            return -fresh_actor_q.mean() + self.actor_bc_coef * fresh_bc_loss

        return AlgorithmOutput(
            loss={"critic": critic_loss, "actor": fresh_actor_loss},
            metrics={
                "ddpg/critic_loss": detached_metric(critic_loss),
                "ddpg/actor_loss": detached_metric(actor_loss),
                "ddpg/actor_bc_loss": detached_metric(actor_bc_loss),
                "ddpg/actor_bc_coef": self.actor_bc_coef,
                "ddpg/q_mean": detached_metric(critic_q.mean()),
                "ddpg/target_mean": detached_metric(target.mean()),
            },
            payload={
                "update_order": ("critic", "actor"),
                "post_step": "ddpg_target",
                "post_step_context": {"tau": self.target_tau},
            },
        )
