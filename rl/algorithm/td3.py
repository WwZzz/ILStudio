"""Twin Delayed Deep Deterministic Policy Gradient."""

from numbers import Real

import torch
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


class TD3Algorithm(BaseRLAlgorithm):
    """Twin critics, smoothed targets, and delayed deterministic actor updates."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        target_tau: float = 0.005,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        actor_bc_coef: float = 0.0,
    ) -> None:
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(target_tau, Real) or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        for name, value in (
            ("target_policy_noise", target_policy_noise),
            ("target_noise_clip", target_noise_clip),
        ):
            if not isinstance(value, Real) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(actor_bc_coef, Real) or float(actor_bc_coef) < 0.0:
            raise ValueError("actor_bc_coef must be non-negative")
        if (
            isinstance(policy_delay, bool)
            or not isinstance(policy_delay, int)
            or policy_delay <= 0
        ):
            raise ValueError("policy_delay must be a positive integer")
        super().__init__(
            required_capabilities=("action", "td3", "target_update"),
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.target_tau = float(target_tau)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.policy_delay = policy_delay
        self.actor_bc_coef = float(actor_bc_coef)
        self._update_step = 0

    @property
    def update_step(self) -> int:
        return self._update_step

    def compute_update(self, batch, *, policy_adapter: BasePolicyAdapter, context=None):
        items = transitions(batch)
        forward_context = {
            **dict(context or {}),
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
        }
        required = (
            "critic_q1",
            "critic_q2",
            "target_next_q1",
            "target_next_q2",
            "actor_q1",
        )
        result = forward_result(
            policy_adapter,
            "td3",
            batch,
            context=forward_context,
            required=required,
        )
        values = {name: vector(result[name], name=name) for name in required}
        if any(len(value) != len(items) for value in values.values()):
            raise ValueError("TD3 outputs must align with transitions")

        target_next_q = torch.minimum(
            values["target_next_q1"], values["target_next_q2"]
        )
        target = decision_rewards(
            items,
            self.reward_key,
            self.gamma,
            like=values["critic_q1"],
        ) + bootstrap_discounts(
            items,
            self.gamma,
            like=values["critic_q1"],
        ) * target_next_q.detach()
        critic1_loss = F.mse_loss(values["critic_q1"], target)
        critic2_loss = F.mse_loss(values["critic_q2"], target)
        critic_loss = critic1_loss + critic2_loss
        actor_bc_loss = result.get("actor_bc_loss")
        if actor_bc_loss is None:
            actor_bc_loss = values["actor_q1"].new_zeros(())
        elif not hasattr(actor_bc_loss, "numel") or actor_bc_loss.numel() != 1:
            raise ValueError("TD3 actor_bc_loss must be a scalar tensor")
        actor_loss = -values["actor_q1"].mean() + self.actor_bc_coef * actor_bc_loss
        actor_due = (self._update_step + 1) % self.policy_delay == 0

        losses = {"critic": critic_loss}
        update_order = ["critic"]
        payload = {"update_order": update_order}
        if actor_due:

            def fresh_actor_loss():
                actor_result = forward_result(
                    policy_adapter,
                    "td3",
                    batch,
                    context={**forward_context, "phase": "actor"},
                    required=("actor_q1",),
                )
                fresh_actor_q1 = vector(
                    actor_result["actor_q1"], name="actor_q1"
                )
                if len(fresh_actor_q1) != len(items):
                    raise ValueError("TD3 actor_q1 must align with transitions")
                fresh_bc_loss = actor_result.get("actor_bc_loss")
                if fresh_bc_loss is None:
                    fresh_bc_loss = fresh_actor_q1.new_zeros(())
                elif not hasattr(fresh_bc_loss, "numel") or fresh_bc_loss.numel() != 1:
                    raise ValueError("TD3 actor_bc_loss must be a scalar tensor")
                return -fresh_actor_q1.mean() + self.actor_bc_coef * fresh_bc_loss

            losses["actor"] = fresh_actor_loss
            update_order.append("actor")
            payload.update(
                {
                    "post_step": "td3_target",
                    "post_step_context": {"tau": self.target_tau},
                }
            )

        return AlgorithmOutput(
            loss=losses,
            metrics={
                "td3/critic1_loss": detached_metric(critic1_loss),
                "td3/critic2_loss": detached_metric(critic2_loss),
                "td3/actor_loss": detached_metric(actor_loss),
                "td3/actor_bc_loss": detached_metric(actor_bc_loss),
                "td3/actor_bc_coef": self.actor_bc_coef,
                "td3/q1_mean": detached_metric(values["critic_q1"].mean()),
                "td3/q2_mean": detached_metric(values["critic_q2"].mean()),
                "td3/target_mean": detached_metric(target.mean()),
                "td3/actor_updated": float(actor_due),
            },
            payload=payload,
        )

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if result.updated:
            self._update_step += 1
        return result

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "update_step": self._update_step,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise TypeError("TD3 algorithm state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported TD3 algorithm state version")
        update_step = state.get("update_step")
        if isinstance(update_step, bool) or not isinstance(update_step, int) or update_step < 0:
            raise ValueError("TD3 update_step must be a non-negative integer")
        self._update_step = update_step
