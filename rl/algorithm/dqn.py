"""Deep Q-Network with optional Double-DQN action selection."""

import copy
from numbers import Real

import torch
import torch.nn.functional as F

from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .utils import (
    bootstrap_masks,
    detached_metric,
    discrete_actions,
    q_matrix,
    rewards,
    transitions,
)


class DQNAlgorithm(BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        double_q: bool = True,
        huber_loss: bool = True,
        target_update_interval: int = 1,
        target_tau: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(double_q, bool) or not isinstance(huber_loss, bool):
            raise TypeError("double_q and huber_loss must be bool")
        if (
            isinstance(target_update_interval, bool)
            or not isinstance(target_update_interval, int)
            or target_update_interval <= 0
        ):
            raise ValueError("target_update_interval must be a positive integer")
        if not isinstance(target_tau, Real) or not 0.0 < float(target_tau) <= 1.0:
            raise ValueError("target_tau must be in (0, 1]")
        if not 0.0 <= float(epsilon_end) <= float(epsilon_start) <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if isinstance(epsilon_decay_steps, bool) or int(epsilon_decay_steps) <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        super().__init__(
            required_capabilities=("action", "action_scores"),
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.double_q = double_q
        self.huber_loss = huber_loss
        self.target_update_interval = target_update_interval
        self.target_tau = float(target_tau)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.update_steps = 0
        self.target_policy = None
        self._pending_target_state = None

    def collection_context(self, context=None):
        env_steps = int(dict(context or {}).get("global_env_steps", 0))
        fraction = min(env_steps / self.epsilon_decay_steps, 1.0)
        epsilon = self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )
        return {"discrete_epsilon": epsilon}

    def _ensure_target_policy(self, policy_adapter):
        if self.target_policy is None:
            self.target_policy = copy.deepcopy(policy_adapter.policy)
            self.target_policy.eval()
            self.target_policy.requires_grad_(False)
            if self._pending_target_state is not None:
                self.target_policy.load_state_dict(self._pending_target_state)
                self._pending_target_state = None
        return self.target_policy

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        target_policy = self._ensure_target_policy(policy_adapter)
        q_values = q_matrix(
            policy_adapter.action_scores(batch, source="obs", context=context),
            name="q_values",
            batch_size=len(items),
        )
        target_next = q_matrix(
            policy_adapter.action_scores(
                batch,
                source="next_obs",
                policy=target_policy,
                context=context,
            ),
            name="target_next_q_values",
            batch_size=len(items),
        )
        actions = discrete_actions(items, device=q_values.device)
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        if self.double_q:
            online_next = q_matrix(
                policy_adapter.action_scores(
                    batch, source="next_obs", context=context
                ),
                name="online_next_q_values",
                batch_size=len(items),
            )
            next_actions = online_next.argmax(dim=1, keepdim=True)
            next_q = target_next.gather(1, next_actions).squeeze(1)
        else:
            next_q = target_next.max(dim=1).values
        target = rewards(items, self.reward_key, like=chosen_q) + (
            self.gamma * bootstrap_masks(items, like=chosen_q) * next_q.detach()
        )
        loss_fn = F.smooth_l1_loss if self.huber_loss else F.mse_loss
        loss = loss_fn(chosen_q, target)
        payload = {}
        if (self.update_steps + 1) % self.target_update_interval == 0:
            payload = {
                "post_step": "dqn_target",
                "post_step_context": {"tau": self.target_tau},
            }
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "dqn/loss": detached_metric(loss),
                "dqn/q_mean": detached_metric(chosen_q.mean()),
                "dqn/target_mean": detached_metric(target.mean()),
            },
            payload=payload,
        )

    def algorithm_post_step(self, operation, *, policy_adapter, context=None):
        if operation != "dqn_target":
            return super().algorithm_post_step(
                operation, policy_adapter=policy_adapter, context=context
            )
        target = self._ensure_target_policy(policy_adapter)
        tau = float(dict(context or {}).get("tau", self.target_tau))
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), policy_adapter.policy.parameters()
            ):
                target_parameter.lerp_(source_parameter, tau)

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if result.updated:
            self.update_steps += 1
        return result

    def state_dict(self):
        state = super().state_dict()
        state["update_steps"] = self.update_steps
        state["target_policy"] = (
            None if self.target_policy is None else self.target_policy.state_dict()
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        update_steps = state.get("update_steps", 0)
        if not isinstance(update_steps, int) or update_steps < 0:
            raise ValueError("DQN update_steps must be a non-negative integer")
        self.update_steps = update_steps
        self._pending_target_state = state.get("target_policy")
