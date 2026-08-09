"""Group Relative Policy Optimization for outcome-reward VLA rollouts."""

from collections import defaultdict
from dataclasses import replace
from numbers import Real

import torch

from rl.policy_adapter import BasePolicyAdapter

from .base import AlgorithmOutput, BaseRLAlgorithm
from .objective import BasePolicyObjectiveBuilder
from .on_policy import FullRolloutUpdates, iter_likelihood_micro_batches
from .utils import detached_metric, forward_result


class GRPOAlgorithm(FullRolloutUpdates, BaseRLAlgorithm):
    """Critic-free clipped policy optimization over groups of trajectories.

    Complete trajectories sharing the same language instruction form a group.
    Their scalar outcome returns are standardized within the group and the
    resulting trajectory advantage is assigned to every policy decision (and,
    through the composed objective builder, every valid action token).
    """

    def __init__(
        self,
        *,
        reward_key: str = "train/total",
        clip_ratio_low: float = 0.2,
        clip_ratio_high: float = 0.28,
        entropy_coef: float = 0.0,
        advantage_epsilon: float = 1e-6,
        update_micro_batch_size: int = 16,
        objective_builder=None,
    ):
        if not isinstance(reward_key, str) or not reward_key:
            raise TypeError("reward_key must be a non-empty string")
        for name, value in (
            ("clip_ratio_low", clip_ratio_low),
            ("clip_ratio_high", clip_ratio_high),
        ):
            if not isinstance(value, Real) or float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(entropy_coef, Real) or float(entropy_coef) < 0:
            raise ValueError("entropy_coef must be non-negative")
        if not isinstance(advantage_epsilon, Real) or float(advantage_epsilon) <= 0:
            raise ValueError("advantage_epsilon must be positive")
        if (
            isinstance(update_micro_batch_size, bool)
            or not isinstance(update_micro_batch_size, int)
            or update_micro_batch_size <= 0
        ):
            raise ValueError("update_micro_batch_size must be a positive integer")
        if not isinstance(objective_builder, BasePolicyObjectiveBuilder):
            raise TypeError("GRPO requires a policy objective builder")
        super().__init__(
            required_capabilities=("action", "grpo"),
            required_buffer_type="rollout",
        )
        self.reward_key = reward_key
        self.clip_ratio_low = float(clip_ratio_low)
        self.clip_ratio_high = float(clip_ratio_high)
        self.entropy_coef = float(entropy_coef)
        self.advantage_epsilon = float(advantage_epsilon)
        self.update_micro_batch_size = update_micro_batch_size
        self.objective_builder = objective_builder

    def _trajectory_records(self, rollout):
        records = {}
        active = {}
        counters = defaultdict(int)
        for step in rollout.steps:
            transition = step.transition
            env_index = int(transition.info.get("env_index", 0))
            trajectory_id = active.setdefault(
                env_index, (env_index, counters[env_index])
            )
            record = records.setdefault(
                trajectory_id,
                {
                    "reward": 0.0,
                    "decision_ids": [],
                    "language": transition.obs.raw_lang,
                    "complete": False,
                },
            )
            value = transition.reward.get(self.reward_key)
            if value is None:
                raise KeyError(
                    f"GRPO transition reward is missing {self.reward_key!r}"
                )
            value = torch.as_tensor(value)
            if value.numel() != 1:
                raise TypeError("GRPO reward must be scalar per environment step")
            record["reward"] += float(value.reshape(-1)[0])
            if step.decision_id not in record["decision_ids"]:
                record["decision_ids"].append(step.decision_id)
            if transition.episode_done:
                record["complete"] = True
                counters[env_index] += 1
                active.pop(env_index, None)
        incomplete = [key for key, record in records.items() if not record["complete"]]
        if incomplete:
            raise ValueError("GRPO requires complete trajectories")
        if len(records) < 2:
            raise ValueError("GRPO requires at least two trajectories per update")
        return tuple(records.values())

    def _decision_advantages(self, rollout, *, like):
        records = self._trajectory_records(rollout)
        grouped = defaultdict(list)
        for record in records:
            grouped[str(record["language"])].append(record)

        by_decision = {}
        active_groups = 0
        rewards = []
        for records_for_prompt in grouped.values():
            if len(records_for_prompt) < 2:
                raise ValueError(
                    "every GRPO language group requires at least two trajectories"
                )
            outcome = like.new_tensor(
                [record["reward"] for record in records_for_prompt]
            )
            rewards.append(outcome)
            std = outcome.std(unbiased=False)
            if float(std.detach()) > self.advantage_epsilon:
                active_groups += 1
            advantage = (outcome - outcome.mean()) / (
                std + self.advantage_epsilon
            )
            for record, value in zip(records_for_prompt, advantage):
                for decision_id in record["decision_ids"]:
                    by_decision[decision_id] = value

        groups = rollout.decision_transitions()
        missing = [
            group.decision.decision_id
            for group in groups
            if group.decision.decision_id not in by_decision
        ]
        if missing:
            raise KeyError(f"GRPO could not assign decisions: {missing!r}")
        return by_decision, torch.cat(rewards), len(grouped), active_groups

    def compute_update(
        self, batch, *, policy_adapter: BasePolicyAdapter, context=None
    ):
        rollout = getattr(batch, "rollout", None)
        source_rollout = getattr(batch, "source_rollout", None) or rollout
        if rollout is None or source_rollout is None:
            raise ValueError("GRPO requires a rollout-aware RolloutBuffer batch")
        result = forward_result(
            policy_adapter,
            "grpo_trace",
            batch,
            context=context,
            required=("traces",),
        )
        first_trace = next(iter(result["traces"].values()))
        like = torch.as_tensor(first_trace.old_logprobs)
        by_decision, outcomes, num_groups, active_groups = self._decision_advantages(
            source_rollout, like=like
        )
        groups = rollout.decision_transitions()
        advantages = torch.stack(
            [by_decision[group.decision.decision_id] for group in groups]
        )
        objective = self.objective_builder.build(
            rollout, result["traces"], advantages
        )
        mask = objective.mask
        if mask is None:
            mask = torch.ones_like(objective.new_logprobs, dtype=torch.bool)
        ratio = objective.ratio
        unclipped = ratio * objective.advantages.detach()
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_ratio_low,
            1.0 + self.clip_ratio_high,
        ) * objective.advantages.detach()
        policy_loss = -torch.minimum(unclipped, clipped).masked_select(mask).mean()
        entropy = result.get("entropy")
        if entropy is None:
            entropy_mean = policy_loss.new_zeros(())
        else:
            entropy_mean = torch.as_tensor(
                entropy, device=policy_loss.device
            ).mean()
        loss = policy_loss - self.entropy_coef * entropy_mean
        log_ratio = (
            objective.new_logprobs - objective.old_logprobs
        ).masked_select(mask)
        clipped_mask = (
            (ratio < 1.0 - self.clip_ratio_low)
            | (ratio > 1.0 + self.clip_ratio_high)
        ).masked_select(mask)
        return AlgorithmOutput(
            loss=loss,
            metrics={
                "grpo/loss": detached_metric(loss),
                "grpo/policy_loss": detached_metric(policy_loss),
                "grpo/entropy": detached_metric(entropy_mean),
                "grpo/approx_kl": detached_metric((-log_ratio).mean()),
                "grpo/clip_fraction": detached_metric(
                    clipped_mask.float().mean()
                ),
                "grpo/outcome_mean": detached_metric(outcomes.mean()),
                "grpo/outcome_std": detached_metric(
                    outcomes.std(unbiased=False)
                ),
                "grpo/num_groups": num_groups,
                "grpo/active_groups": active_groups,
                "grpo/num_trajectories": int(outcomes.numel()),
                "grpo/num_objectives": self._num_trace_objectives(source_rollout),
                "grpo/micro_num_objectives": int(mask.sum()),
            },
        )

    @staticmethod
    def _num_trace_objectives(rollout):
        total = 0
        for decision in rollout.decisions:
            trace = decision.trace
            if trace is None:
                raise ValueError("GRPO decisions require stored policy traces")
            logprobs = torch.as_tensor(trace.old_logprobs)
            if trace.valid_mask is None:
                total += logprobs.numel()
            else:
                total += int(torch.as_tensor(trace.valid_mask).sum())
        if total <= 0:
            raise ValueError("GRPO rollout has no valid policy objectives")
        return total

    def iter_compute_updates(
        self,
        batch,
        *,
        policy_adapter: BasePolicyAdapter,
        context=None,
    ):
        rollout = getattr(batch, "rollout", None)
        if rollout is None:
            raise ValueError("GRPO requires a rollout-aware RolloutBuffer batch")
        decisions = tuple(rollout.decisions)
        total_objectives = self._num_trace_objectives(rollout)
        for selected in iter_likelihood_micro_batches(
            decisions, self.update_micro_batch_size
        ):
            micro_batch = batch.select_decisions(
                decision.decision_id for decision in selected
            )
            output = self.compute_update(
                micro_batch,
                policy_adapter=policy_adapter,
                context=context,
            )
            micro_objectives = output.metrics["grpo/micro_num_objectives"]
            weight = float(micro_objectives) / float(total_objectives)
            payload = dict(output.payload)
            payload["loss_weight"] = weight
            payload["metric_weight"] = weight
            yield replace(output, payload=payload)
