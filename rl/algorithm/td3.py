"""Twin Delayed DDPG with algorithm-owned target policy and target critics."""

from numbers import Real

import torch
import torch.nn.functional as F

from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput
from .ddpg import DDPGAlgorithm, _action_batch, _masked_action_mse
from .utils import (
    bootstrap_discounts,
    decision_rewards,
    detached_metric,
    transitions,
    validate_policy_result,
    vector,
)


def _chunk_ar1_noise_like(reference, *, std, rho, clip):
    """Sample stationary correlated target noise for action chunks."""

    if not torch.is_tensor(reference) or reference.ndim != 3:
        raise ValueError("chunk noise reference must be rank three")
    for name, value in (("std", std), ("rho", rho)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
    if float(std) < 0.0:
        raise ValueError("std must be non-negative")
    if not 0.0 <= float(rho) < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if clip is not None and float(clip) <= 0.0:
        raise ValueError("clip must be positive or None")
    noise = torch.randn_like(reference) * float(std)
    innovation_scale = (1.0 - float(rho) ** 2) ** 0.5
    for index in range(1, reference.shape[1]):
        noise[:, index] = (
            float(rho) * noise[:, index - 1]
            + innovation_scale * noise[:, index]
        )
    return noise if clip is None else noise.clamp(-float(clip), float(clip))


class TD3Algorithm(DDPGAlgorithm):
    METRIC_PREFIX = "td3"

    def __init__(
        self,
        *,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        target_policy_noise_rho: float = 0.0,
        policy_delay: int = 2,
        **kwargs,
    ):
        for name, value in (
            ("target_policy_noise", target_policy_noise),
            ("target_noise_clip", target_noise_clip),
        ):
            if not isinstance(value, Real) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(target_policy_noise_rho, Real) or not 0.0 <= float(target_policy_noise_rho) < 1.0:
            raise ValueError("target_policy_noise_rho must be in [0, 1)")
        if isinstance(policy_delay, bool) or not isinstance(policy_delay, int) or policy_delay <= 0:
            raise ValueError("policy_delay must be a positive integer")
        super().__init__(**kwargs)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.target_policy_noise_rho = float(target_policy_noise_rho)
        self.policy_delay = policy_delay
        self._update_step = 0

    @property
    def update_step(self):
        return self._update_step

    def _twin_values(self, value, *, prefix):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("TD3 requires a twin-Q critic")
        return vector(value[0], name=f"{prefix}1"), vector(
            value[1], name=f"{prefix}2"
        )

    def _target_noise(self, actions):
        if actions.ndim == 3:
            return _chunk_ar1_noise_like(
                actions,
                std=self.target_policy_noise,
                rho=self.target_policy_noise_rho,
                clip=self.target_noise_clip,
            )
        noise = torch.randn_like(actions) * self.target_policy_noise
        return noise.clamp(-self.target_noise_clip, self.target_noise_clip)

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
        actor_q1, _ = self._twin_values(
            self.critic(
                observations,
                actor["action"],
                action_mask=action_mask,
                context=context,
            ),
            prefix="actor_q",
        )
        bc_loss = _masked_action_mse(actor["action"], replay_actions, action_mask)
        return self._combine_actor_loss(actor_q1, bc_loss), actor_q1, bc_loss

    def _combine_actor_loss(self, actor_q1, bc_loss):
        return -actor_q1.mean() + self.actor_bc_coef * bc_loss

    def _extra_actor_metrics(self, actor_q1):
        del actor_q1
        return {}

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        replay_actions, action_mask = _action_batch(
            policy_adapter.batch_actions(batch, context=context)
        )
        critic_q1, critic_q2 = self._twin_values(
            self.critic(
                observations,
                replay_actions,
                action_mask=action_mask,
                context=context,
            ),
            prefix="critic_q",
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
            noise = self._target_noise(target_actor["action"])
            target_action = policy_adapter.clamp_actions(target_actor["action"] + noise)
            target_q1, target_q2 = self._twin_values(
                self.critic.target(
                    next_observations,
                    target_action,
                    context={**dict(context or {}), "feature_policy": target_policy},
                ),
                prefix="target_q",
            )
        target = decision_rewards(
            items, self.reward_key, self.gamma, like=critic_q1
        ) + bootstrap_discounts(items, self.gamma, like=critic_q1) * torch.minimum(
            target_q1, target_q2
        )
        critic_loss = F.mse_loss(critic_q1, target.detach()) + F.mse_loss(
            critic_q2, target.detach()
        )
        update_actor = (self._update_step + 1) % self.policy_delay == 0
        losses = {"critic": critic_loss}
        order = ["critic"]
        prefix = self.METRIC_PREFIX
        metrics = {
            f"{prefix}/critic_loss": detached_metric(critic_loss),
            f"{prefix}/target_mean": detached_metric(target.mean()),
            f"{prefix}/actor_updated": float(update_actor),
        }
        payload = {"update_order": tuple(order)}
        if update_actor:
            actor_loss, actor_q1, bc_loss = self._actor_loss(
                batch, items, policy_adapter, context
            )

            def fresh_actor_loss():
                return self._actor_loss(batch, items, policy_adapter, context)[0]

            losses["actor"] = fresh_actor_loss
            order.append("actor")
            payload.update(
                {
                    "update_order": tuple(order),
                    "post_step": "target_networks",
                    "post_step_context": {"tau": self.target_tau},
                }
            )
            metrics.update(
                {
                    f"{prefix}/actor_loss": detached_metric(actor_loss),
                    f"{prefix}/actor_q_mean": detached_metric(actor_q1.mean()),
                    f"{prefix}/actor_bc_loss": detached_metric(bc_loss),
                }
            )
            metrics.update(self._extra_actor_metrics(actor_q1))
        return AlgorithmOutput(loss=losses, metrics=metrics, payload=payload)

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if result.updated:
            self._update_step += 1
        return result

    def state_dict(self):
        state = super().state_dict()
        state["update_step"] = self._update_step
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._update_step = int(state.get("update_step", 0))
