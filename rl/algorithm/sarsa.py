"""On-policy SARSA temporal-difference control."""

from numbers import Real

import torch
import torch.nn.functional as F

from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .on_policy import FullRolloutUpdates
from .utils import (
    bootstrap_masks,
    detached_metric,
    discrete_actions,
    q_matrix,
    rewards,
    transitions,
)


class SARSAAlgorithm(FullRolloutUpdates, BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        huber_loss: bool = True,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(huber_loss, bool):
            raise TypeError("huber_loss must be bool")
        if not 0.0 <= float(epsilon_end) <= float(epsilon_start) <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if isinstance(epsilon_decay_steps, bool) or int(epsilon_decay_steps) <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        super().__init__(
            required_capabilities=("action", "action_scores"),
            required_buffer_type="rollout",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.huber_loss = huber_loss
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

    def collection_context(self, context=None):
        env_steps = int(dict(context or {}).get("global_env_steps", 0))
        fraction = min(env_steps / self.epsilon_decay_steps, 1.0)
        epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )
        return {"discrete_epsilon": epsilon}

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        q_values = q_matrix(
            policy_adapter.action_scores(batch, source="obs", context=context),
            name="q_values",
            batch_size=len(items),
        )
        next_q_values = q_matrix(
            policy_adapter.action_scores(batch, source="next_obs", context=context),
            name="next_q_values",
            batch_size=len(items),
        )
        actions = discrete_actions(items, device=q_values.device)
        derived = []
        for index, item in enumerate(items):
            if item.terminated:
                derived.append(0)
            elif not item.episode_done and index + 1 < len(items):
                next_action = discrete_actions((items[index + 1],), device=q_values.device)
                derived.append(int(next_action[0]))
            else:
                raise KeyError(
                    "SARSA needs a following on-policy action for rollout tails"
                )
        next_actions = torch.as_tensor(derived, device=q_values.device, dtype=torch.long)
        if len(next_actions) != len(items):
            raise ValueError("next_action must contain one action per transition")
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        next_q = next_q_values.gather(1, next_actions[:, None]).squeeze(1)
        target = rewards(items, self.reward_key, like=chosen_q) + (
            self.gamma * bootstrap_masks(items, like=chosen_q) * next_q.detach()
        )
        loss_fn = F.smooth_l1_loss if self.huber_loss else F.mse_loss
        loss = loss_fn(chosen_q, target)
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "sarsa/loss": detached_metric(loss),
                "sarsa/q_mean": detached_metric(chosen_q.mean()),
                "sarsa/target_mean": detached_metric(target.mean()),
            },
        )
