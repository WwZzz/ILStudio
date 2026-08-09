"""Soft Actor-Critic with twin Q critics and optional temperature learning."""

from numbers import Real

import torch
import torch.nn.functional as F

from rl.policy_adapter import BasePolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .utils import (
    bootstrap_masks,
    detached_metric,
    forward_result,
    rewards,
    transitions,
    vector,
)


class SACAlgorithm(BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        alpha=0.2,
        target_entropy: float = -1.0,
        target_tau: float = 0.005,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if alpha != "auto" and (
            not isinstance(alpha, Real) or float(alpha) <= 0.0
        ):
            raise ValueError("alpha must be positive or 'auto'")
        if not isinstance(target_entropy, Real):
            raise TypeError("target_entropy must be numeric")
        if not isinstance(target_tau, Real) or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        capabilities = {"action", "sac", "target_update"}
        if alpha == "auto":
            capabilities.add("temperature")
        super().__init__(
            required_capabilities=capabilities,
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.alpha = alpha if alpha == "auto" else float(alpha)
        self.target_entropy = float(target_entropy)
        self.target_tau = float(target_tau)

    def _temperature(self, result, *, like):
        if self.alpha != "auto":
            return like.new_tensor(self.alpha), None
        if "log_alpha" not in result:
            raise KeyError("auto-temperature SAC requires log_alpha")
        log_alpha = result["log_alpha"]
        if not torch.is_tensor(log_alpha):
            raise TypeError("log_alpha must be a trainable tensor")
        return log_alpha.exp(), log_alpha

    def compute_update(self, batch, *, policy_adapter: BasePolicyAdapter, context=None):
        items = transitions(batch)
        required = (
            "q1",
            "q2",
            "target_next_q1",
            "target_next_q2",
            "next_log_prob",
            "actor_q1",
            "actor_q2",
            "actor_log_prob",
        )
        result = forward_result(
            policy_adapter,
            "sac",
            batch,
            context=context,
            required=required,
        )
        values = {
            name: vector(result[name], name=name)
            for name in required
        }
        if any(len(value) != len(items) for value in values.values()):
            raise ValueError("SAC outputs must align with transitions")
        alpha, log_alpha = self._temperature(result, like=values["q1"])
        target_next_q = torch.minimum(
            values["target_next_q1"], values["target_next_q2"]
        ) - alpha.detach() * values["next_log_prob"]
        target = rewards(items, self.reward_key, like=values["q1"]) + (
            self.gamma
            * bootstrap_masks(items, like=values["q1"])
            * target_next_q.detach()
        )
        critic1_loss = F.mse_loss(values["q1"], target)
        critic2_loss = F.mse_loss(values["q2"], target)
        actor_loss = (
            alpha.detach() * values["actor_log_prob"]
            - torch.minimum(values["actor_q1"], values["actor_q2"])
        ).mean()

        def fresh_actor_loss():
            actor_result = forward_result(
                policy_adapter,
                "sac",
                batch,
                context={**dict(context or {}), "phase": "actor"},
                required=("actor_q1", "actor_q2", "actor_log_prob"),
            )
            actor_q1 = vector(actor_result["actor_q1"], name="actor_q1")
            actor_q2 = vector(actor_result["actor_q2"], name="actor_q2")
            log_prob = vector(actor_result["actor_log_prob"], name="actor_log_prob")
            if any(len(value) != len(items) for value in (actor_q1, actor_q2, log_prob)):
                raise ValueError("SAC actor outputs must align with transitions")
            return (alpha.detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()

        losses = {
            "critic1": critic1_loss,
            "critic2": critic2_loss,
            "actor": fresh_actor_loss,
        }
        update_order = ["critic1", "critic2", "actor"]
        metrics = {
            "sac/critic1_loss": detached_metric(critic1_loss),
            "sac/critic2_loss": detached_metric(critic2_loss),
            "sac/actor_loss": detached_metric(actor_loss),
            "sac/alpha": detached_metric(alpha.mean()),
            "sac/target_mean": detached_metric(target.mean()),
        }
        if log_alpha is not None:
            alpha_loss = -(
                log_alpha * (values["actor_log_prob"].detach() + self.target_entropy)
            ).mean()
            losses["alpha"] = alpha_loss
            update_order.append("alpha")
            metrics["sac/alpha_loss"] = detached_metric(alpha_loss)
        return AlgorithmOutput(
            loss=losses,
            metrics=metrics,
            payload={
                "update_order": tuple(update_order),
                "post_step": "sac_target",
                "post_step_context": {"tau": self.target_tau},
            },
        )
