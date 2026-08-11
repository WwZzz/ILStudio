"""Conservative Q-Learning built as a SAC critic regularizer."""

from numbers import Real

import torch

from .sac import SACAlgorithm
from .utils import detached_metric, validate_policy_result, vector


class CQLAlgorithm(SACAlgorithm):
    """Continuous-action CQL(H) with random and policy action samples."""

    METRIC_PREFIX = "cql"

    def __init__(
        self,
        *,
        conservative_weight=1.0,
        conservative_temperature=1.0,
        num_action_samples=10,
        **kwargs,
    ):
        for name, value in (
            ("conservative_weight", conservative_weight),
            ("conservative_temperature", conservative_temperature),
        ):
            if not isinstance(value, Real) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if isinstance(num_action_samples, bool) or not isinstance(
            num_action_samples, int
        ):
            raise TypeError("num_action_samples must be an integer")
        if num_action_samples <= 0:
            raise ValueError("num_action_samples must be positive")
        super().__init__(**kwargs)
        self.required_capabilities = frozenset(
            set(self.required_capabilities) | {"uniform_actions"}
        )
        self.conservative_weight = float(conservative_weight)
        self.conservative_temperature = float(conservative_temperature)
        self.num_action_samples = num_action_samples

    @staticmethod
    def _sample_q_values(
        critic,
        observations,
        actions,
        log_prob,
        *,
        context,
        prefix,
    ):
        q1_values = []
        q2_values = []
        for index in range(actions.shape[0]):
            q1, q2 = critic(observations, actions[index], context=context)
            q1_values.append(vector(q1, name=f"{prefix}_q1") - log_prob[index])
            q2_values.append(vector(q2, name=f"{prefix}_q2") - log_prob[index])
        return torch.stack(q1_values), torch.stack(q2_values)

    def critic_regularization(
        self,
        batch,
        *,
        observations,
        next_observations,
        replay_actions,
        q1,
        q2,
        policy_adapter,
        context,
    ):
        del replay_actions
        uniform = validate_policy_result(
            policy_adapter.uniform_actions(
                batch,
                num_samples=self.num_action_samples,
                context=context,
            ),
            operation="uniform_actions",
            required=("action", "log_prob"),
        )
        uniform_actions = uniform["action"]
        uniform_log_prob = uniform["log_prob"]
        if not torch.is_tensor(uniform_actions) or uniform_actions.ndim < 3:
            raise ValueError("uniform actions must have [sample, batch, ...] shape")
        if not torch.is_tensor(uniform_log_prob) or uniform_log_prob.shape != uniform_actions.shape[:2]:
            raise ValueError("uniform log_prob must have shape [sample, batch]")
        random_q1, random_q2 = self._sample_q_values(
            self.critic,
            observations,
            uniform_actions,
            uniform_log_prob,
            context=context,
            prefix="random",
        )

        current_q1 = []
        current_q2 = []
        next_q1 = []
        next_q2 = []
        for _ in range(self.num_action_samples):
            current = validate_policy_result(
                policy_adapter.sample_actions(
                    batch,
                    source="obs",
                    context=context,
                ),
                operation="sample_actions",
                required=("action", "log_prob"),
            )
            following = validate_policy_result(
                policy_adapter.sample_actions(
                    batch,
                    source="next_obs",
                    context=context,
                ),
                operation="sample_actions",
                required=("action", "log_prob"),
            )
            cq1, cq2 = self.critic(
                observations, current["action"].detach(), context=context
            )
            nq1, nq2 = self.critic(
                observations, following["action"].detach(), context=context
            )
            current_log_prob = vector(
                current["log_prob"], name="current_log_prob"
            ).detach()
            next_log_prob = vector(
                following["log_prob"], name="next_log_prob"
            ).detach()
            current_q1.append(vector(cq1, name="current_q1") - current_log_prob)
            current_q2.append(vector(cq2, name="current_q2") - current_log_prob)
            next_q1.append(vector(nq1, name="next_q1") - next_log_prob)
            next_q2.append(vector(nq2, name="next_q2") - next_log_prob)

        temperature = self.conservative_temperature
        candidates1 = torch.cat(
            (random_q1, torch.stack(current_q1), torch.stack(next_q1)), dim=0
        )
        candidates2 = torch.cat(
            (random_q2, torch.stack(current_q2), torch.stack(next_q2)), dim=0
        )
        gap1 = temperature * torch.logsumexp(
            candidates1 / temperature, dim=0
        ).mean() - q1.mean()
        gap2 = temperature * torch.logsumexp(
            candidates2 / temperature, dim=0
        ).mean() - q2.mean()
        regularizer1 = self.conservative_weight * gap1
        regularizer2 = self.conservative_weight * gap2
        return regularizer1, regularizer2, {
            "cql/conservative1_loss": detached_metric(regularizer1),
            "cql/conservative2_loss": detached_metric(regularizer2),
            "cql/q1_gap": detached_metric(gap1),
            "cql/q2_gap": detached_metric(gap2),
        }


__all__ = ["CQLAlgorithm"]
