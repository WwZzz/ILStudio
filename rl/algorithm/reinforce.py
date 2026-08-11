"""Monte-Carlo REINFORCE policy-gradient algorithm."""

from numbers import Real

from rl.policy_adapter import MetaPolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .on_policy import FullRolloutUpdates
from .utils import (
    detached_metric,
    discounted_returns,
    normalized,
    rewards,
    transitions,
    validate_policy_result,
    vector,
)


class ReinforceAlgorithm(FullRolloutUpdates, BaseRLAlgorithm):
    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        normalize_returns: bool = True,
        entropy_coef: float = 0.0,
    ):
        if not isinstance(gamma, Real) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        if not isinstance(normalize_returns, bool):
            raise TypeError("normalize_returns must be bool")
        if not isinstance(entropy_coef, Real) or float(entropy_coef) < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        super().__init__(
            required_capabilities=("action", "evaluate_actions"),
            required_buffer_type="rollout",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.normalize_returns = normalize_returns
        self.entropy_coef = float(entropy_coef)

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        items = transitions(batch)
        result = validate_policy_result(
            policy_adapter.evaluate_actions(batch, context=context),
            operation="evaluate_actions",
            required=("log_prob",),
        )
        log_prob = vector(result["log_prob"], name="log_prob")
        if len(log_prob) != len(items):
            raise ValueError("log_prob length must match transitions")
        returns = discounted_returns(
            items,
            rewards(items, self.reward_key, like=log_prob),
            self.gamma,
        )
        weights = normalized(returns) if self.normalize_returns else returns
        entropy = vector(result.get("entropy", log_prob.new_zeros(len(items))), name="entropy")
        loss = -(log_prob * weights.detach()).mean() - self.entropy_coef * entropy.mean()
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "reinforce/loss": detached_metric(loss),
                "reinforce/return_mean": detached_metric(returns.mean()),
                "reinforce/entropy": detached_metric(entropy.mean()),
            },
        )
