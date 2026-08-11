"""Soft Actor-Critic with twin Q critics and optional temperature learning."""

from numbers import Real

import torch
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


class SACAlgorithm(BaseRLAlgorithm):
    METRIC_PREFIX = "sac"

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        reward_key: str = "train/total",
        alpha=0.2,
        target_entropy: float = -1.0,
        target_tau: float = 0.005,
        critic=None,
        actor_learning_rate=None,
        critic_learning_rate=None,
        alpha_learning_rate=None,
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
        if not isinstance(critic, BaseCritic):
            raise TypeError("SAC requires a configured critic inheriting BaseCritic")
        for name, value in (
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
            ("alpha_learning_rate", alpha_learning_rate),
        ):
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("target", "parameter_groups", "soft_update"):
            if not callable(getattr(critic, name, None)):
                raise TypeError(f"SAC critic must provide {name}()")
        capabilities = {"action", "batch_actions", "sample_actions"}
        super().__init__(
            required_capabilities=capabilities,
            required_buffer_type="replay",
        )
        self.gamma = float(gamma)
        self.reward_key = reward_key
        self.alpha = alpha if alpha == "auto" else float(alpha)
        self.target_entropy = float(target_entropy)
        self.target_tau = float(target_tau)
        self.critic = critic
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.log_alpha = None
        if alpha == "auto":
            device = next(critic.parameters()).device
            self.log_alpha = torch.nn.Parameter(torch.zeros((), device=device))

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
        del (
            batch,
            observations,
            next_observations,
            replay_actions,
            q1,
            q2,
            policy_adapter,
            context,
        )
        return None, None, {}

    def _temperature(self, result, *, like):
        if self.alpha != "auto":
            return like.new_tensor(self.alpha), None
        log_alpha = self.log_alpha
        if log_alpha is None:
            if "log_alpha" not in result:
                raise KeyError("auto-temperature SAC requires log_alpha")
            log_alpha = result["log_alpha"]
        if not torch.is_tensor(log_alpha):
            raise TypeError("log_alpha must be a trainable tensor")
        return log_alpha.exp(), log_alpha

    @staticmethod
    def _actor_parameters(policy_adapter):
        parameter_adapter = getattr(policy_adapter, "actor_parameters", None)
        if callable(parameter_adapter):
            return tuple(parameter_adapter())
        parameters = []
        seen = set()
        for owner in (policy_adapter.policy, policy_adapter):
            for parameter in owner.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    def parameters(self):
        parameters = list(self.critic.parameters())
        if self.log_alpha is not None:
            parameters.append(self.log_alpha)
        return tuple(parameter for parameter in parameters if parameter.requires_grad)

    def optimizer_parameter_groups(self, policy_adapter):
        groups = {
            name: {
                "parameters": parameters,
                "learning_rate": self.critic_learning_rate,
            }
            for name, parameters in self.critic.parameter_groups().items()
        }
        groups["actor"] = {
            "parameters": self._actor_parameters(policy_adapter),
            "learning_rate": self.actor_learning_rate,
        }
        if self.log_alpha is not None:
            groups["alpha"] = {
                "parameters": (self.log_alpha,),
                "learning_rate": self.alpha_learning_rate,
            }
        return groups

    def _compute_composed_update(self, batch, *, policy_adapter, context):
        items = transitions(batch)
        observations = tuple(item.obs for item in items)
        next_observations = tuple(item.next_obs for item in items)
        replay_actions = policy_adapter.batch_actions(batch, context=context)
        actor = validate_policy_result(
            policy_adapter.sample_actions(
                batch,
                source="obs",
                context=context,
            ),
            operation="sample_actions",
            required=("action", "log_prob"),
        )
        q1, q2 = self.critic(
            observations,
            replay_actions,
            context=context,
        )
        actor_q1, actor_q2 = self.critic(
            observations,
            actor["action"],
            context=context,
        )
        with torch.no_grad():
            next_actor = validate_policy_result(
                policy_adapter.sample_actions(
                    batch,
                    source="next_obs",
                    context=context,
                ),
                operation="sample_actions",
                required=("action", "log_prob"),
            )
            target_next_q1, target_next_q2 = self.critic.target(
                next_observations,
                next_actor["action"],
                context=context,
            )
        values = {
            "q1": vector(q1, name="q1"),
            "q2": vector(q2, name="q2"),
            "target_next_q1": vector(target_next_q1, name="target_next_q1"),
            "target_next_q2": vector(target_next_q2, name="target_next_q2"),
            "next_log_prob": vector(next_actor["log_prob"], name="next_log_prob"),
            "actor_q1": vector(actor_q1, name="actor_q1"),
            "actor_q2": vector(actor_q2, name="actor_q2"),
            "actor_log_prob": vector(actor["log_prob"], name="actor_log_prob"),
        }
        alpha, log_alpha = self._temperature({}, like=values["q1"])
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
        regularizer1, regularizer2, regularizer_metrics = self.critic_regularization(
            batch,
            observations=observations,
            next_observations=next_observations,
            replay_actions=replay_actions,
            q1=values["q1"],
            q2=values["q2"],
            policy_adapter=policy_adapter,
            context=context,
        )
        if regularizer1 is not None:
            critic1_loss = critic1_loss + regularizer1
        if regularizer2 is not None:
            critic2_loss = critic2_loss + regularizer2
        actor_loss = (
            alpha.detach() * values["actor_log_prob"]
            - torch.minimum(values["actor_q1"], values["actor_q2"])
        ).mean()

        def fresh_actor_loss():
            fresh_actor = validate_policy_result(
                policy_adapter.sample_actions(
                    batch,
                    source="obs",
                    context={**dict(context or {}), "phase": "actor"},
                ),
                operation="sample_actions",
                required=("action", "log_prob"),
            )
            fresh_q1, fresh_q2 = self.critic(
                observations,
                fresh_actor["action"],
                context=context,
            )
            return (
                alpha.detach()
                * vector(fresh_actor["log_prob"], name="actor_log_prob")
                - torch.minimum(
                    vector(fresh_q1, name="actor_q1"),
                    vector(fresh_q2, name="actor_q2"),
                )
            ).mean()

        losses = {
            "critic1": critic1_loss,
            "critic2": critic2_loss,
            "actor": fresh_actor_loss,
        }
        update_order = ["critic1", "critic2", "actor"]
        prefix = self.METRIC_PREFIX
        metrics = {
            f"{prefix}/critic1_loss": detached_metric(critic1_loss),
            f"{prefix}/critic2_loss": detached_metric(critic2_loss),
            f"{prefix}/actor_loss": detached_metric(actor_loss),
            f"{prefix}/alpha": detached_metric(alpha.mean()),
            f"{prefix}/target_mean": detached_metric(target.mean()),
        }
        metrics.update(regularizer_metrics)
        if log_alpha is not None:
            alpha_loss = -(
                log_alpha
                * (values["actor_log_prob"].detach() + self.target_entropy)
            ).mean()
            losses["alpha"] = alpha_loss
            update_order.append("alpha")
            metrics[f"{prefix}/alpha_loss"] = detached_metric(alpha_loss)
        return AlgorithmOutput(
            loss=losses,
            metrics=metrics,
            payload={
                "update_order": tuple(update_order),
                "post_step": "sac_target",
                "post_step_context": {"tau": self.target_tau},
            },
        )

    def compute_update(self, batch, *, policy_adapter: MetaPolicyAdapter, context=None):
        return self._compute_composed_update(
            batch,
            policy_adapter=policy_adapter,
            context=context,
        )

    def algorithm_post_step(self, operation, *, policy_adapter, context=None):
        if operation == "sac_target":
            self.critic.soft_update(dict(context or {}).get("tau", self.target_tau))
            return
        super().algorithm_post_step(
            operation,
            policy_adapter=policy_adapter,
            context=context,
        )

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "critic": self.critic.state_dict(),
                "log_alpha": (
                    None
                    if self.log_alpha is None
                    else self.log_alpha.detach().cpu()
                ),
            }
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        critic_state = state.get("critic")
        if critic_state is not None:
            self.critic.load_state_dict(critic_state)
        saved_log_alpha = state.get("log_alpha")
        if (saved_log_alpha is None) != (self.log_alpha is None):
            raise ValueError("SAC temperature state does not match configuration")
        if self.log_alpha is not None:
            self.log_alpha.data.copy_(saved_log_alpha.to(self.log_alpha.device))
