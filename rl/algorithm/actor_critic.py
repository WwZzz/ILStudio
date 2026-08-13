"""One-step advantage actor-critic algorithm."""

from numbers import Real

import torch.nn.functional as F

from rl.policy_adapter import MetaPolicyAdapter
from rl.critic import BaseCritic

from .base import AlgorithmOutput, BaseRLAlgorithm
from .utils import (
    bootstrap_masks,
    detached_metric,
    rewards,
    transitions,
    validate_policy_result,
    vector,
)


class ActorCriticAlgorithm(BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        critic=None,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(value_coef, Real) or float(value_coef) < 0.0:
            raise ValueError("value_coef must be non-negative")
        if not isinstance(entropy_coef, Real) or float(entropy_coef) < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if not isinstance(critic, BaseCritic):
            raise TypeError("actor-critic requires a configured BaseCritic")
        super().__init__(
            required_capabilities=("action", "evaluate_actions"),
            required_buffer_type="rollout",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.critic = critic

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        result = validate_policy_result(
            policy_adapter.evaluate_actions(batch, context=context),
            operation="evaluate_actions",
            required=("log_prob",),
        )
        log_prob = vector(result["log_prob"], name="log_prob")
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        value = vector(self.critic(observations, context=context), name="value")
        next_value = vector(
            self.critic(next_observations, context=context), name="next_value"
        )
        if any(len(item) != len(items) for item in (log_prob, value, next_value)):
            raise ValueError("actor-critic outputs must align with transitions")
        reward = rewards(items, self.reward_key, like=value)
        target = reward + self.gamma * bootstrap_masks(items, like=value) * next_value.detach()
        advantage = target - value
        entropy = vector(result.get("entropy", value.new_zeros(len(items))), name="entropy")
        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = F.mse_loss(value, target)
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "actor_critic/loss": detached_metric(loss),
                "actor_critic/actor_loss": detached_metric(actor_loss),
                "actor_critic/critic_loss": detached_metric(critic_loss),
                "actor_critic/advantage_mean": detached_metric(advantage.mean()),
            },
        )

    def parameters(self):
        return tuple(self.critic.parameters())

    def state_dict(self):
        state = super().state_dict()
        state["critic"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.critic.load_state_dict(state["critic"])
