"""Synchronous collector over an ILStudio benchmark environment runner."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Optional

from benchmark.env_runner import BaseEnvRunner
from rl.base import (
    RL_TERMINATED_ON_SUCCESS_KEY,
    MetaTransition,
    Rollout,
    RolloutStep,
)
from rl.buffer import BaseBuffer
from rl.executor import BasePolicyExecutor
from rl.reward import RewardComposer, wrap_env_reward

from .base import (
    BaseCollector,
    CollectResult,
    EpisodeSummary,
    annotate_episode_timestep,
    apply_episode_semantics,
    validate_episode_semantics,
)


def _validate_target(value, *, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class SyncCollector(BaseCollector):
    """Coordinate runner, executor, reward composition and buffer writes."""

    def __init__(
        self,
        *,
        runner: BaseEnvRunner,
        executor: BasePolicyExecutor,
        buffer: Optional[BaseBuffer] = None,
        reward_composer: Optional[RewardComposer] = None,
        terminate_on_success: bool = False,
        bootstrap_on_truncation: bool = True,
    ) -> None:
        if not isinstance(runner, BaseEnvRunner):
            raise TypeError("runner must inherit BaseEnvRunner")
        if runner.num_envs != 1:
            raise ValueError("SyncCollector requires a single-environment runner")
        if not isinstance(executor, BasePolicyExecutor):
            raise TypeError("executor must inherit BasePolicyExecutor")
        if buffer is not None and not isinstance(buffer, BaseBuffer):
            raise TypeError("buffer must inherit BaseBuffer")
        if reward_composer is not None and not isinstance(
            reward_composer, RewardComposer
        ):
            raise TypeError("reward_composer must be RewardComposer")
        self.runner = runner
        self.executor = executor
        self.buffer = buffer
        self.reward_composer = reward_composer or RewardComposer()
        (
            self.terminate_on_success,
            self.bootstrap_on_truncation,
        ) = validate_episode_semantics(
            terminate_on_success=terminate_on_success,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )
        self._obs = None
        self._episode_length = 0
        self._episode_reward = {}
        self._episode_success = False
        self._wrapped_reset_pending = False
        self._episode_index = 0
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("collector is closed")

    def _ensure_episode(self):
        if self._obs is None:
            if not self.runner.needs_reset and not self._wrapped_reset_pending:
                raise RuntimeError("collector lost observation for an active environment")
            self.executor.reset()
            self._obs = annotate_episode_timestep(
                self.runner.reset(), 0
            )
            self._wrapped_reset_pending = False
            self._episode_length = 0
            self._episode_reward = {}
            self._episode_success = False

    def _accumulate_reward(self, reward):
        for key, value in reward.items():
            if key in self._episode_reward:
                self._episode_reward[key] = self._episode_reward[key] + value
            else:
                self._episode_reward[key] = value

    def _finish_episode(self, transition):
        summary = EpisodeSummary(
            index=self._episode_index,
            length=self._episode_length,
            reward=self._episode_reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            success=self._episode_success,
            info=transition.info,
        )
        self._episode_index += 1
        self._obs = None
        self._episode_length = 0
        self._episode_reward = {}
        self._episode_success = False
        self._wrapped_reset_pending = bool(
            transition.info.get(RL_TERMINATED_ON_SUCCESS_KEY, False)
        )
        self.executor.reset()
        return summary

    def collect(
        self,
        *,
        num_steps=None,
        num_episodes=None,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> CollectResult:
        self._ensure_open()
        if (num_steps is None) == (num_episodes is None):
            raise ValueError("provide exactly one of num_steps or num_episodes")
        if num_steps is not None:
            _validate_target(num_steps, name="num_steps")
        if num_episodes is not None:
            _validate_target(num_episodes, name="num_episodes")

        transitions = []
        episodes = []
        rollout_steps = []
        decisions = {}
        provenance_mode = None
        while True:
            if num_steps is not None and len(transitions) >= num_steps:
                break
            if num_episodes is not None and len(episodes) >= num_episodes:
                break

            self._ensure_episode()
            output = self.executor.select_action(
                self._obs,
                deterministic=deterministic,
                context=context,
            )
            next_obs, env_reward, terminated, truncated, info = self.runner.step(
                output.action
            )
            next_obs = annotate_episode_timestep(
                next_obs, self._episode_length + 1
            )
            transition = MetaTransition(
                obs=self._obs,
                action=output.action,
                next_obs=next_obs,
                reward=wrap_env_reward(env_reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                policy_info=output.policy_info,
            )
            transition = apply_episode_semantics(
                transition,
                terminate_on_success=self.terminate_on_success,
                bootstrap_on_truncation=self.bootstrap_on_truncation,
            )
            reward = self.reward_composer.compute_step(
                transition,
                context=context,
            )
            transition = replace(transition, reward=reward)
            transitions.append(transition)
            has_decision = output.decision is not None
            if provenance_mode is None:
                provenance_mode = has_decision
            elif provenance_mode != has_decision:
                raise RuntimeError(
                    "executor cannot mix decision-aware and legacy outputs"
                )
            rollout_step = None
            if has_decision:
                rollout_step = RolloutStep(
                    transition=transition,
                    decision_id=output.decision.decision_id,
                    action_offset=output.action_offset,
                )
                existing = decisions.get(output.decision.decision_id)
                if existing is not None and existing is not output.decision:
                    raise ValueError("one decision_id refers to different decisions")
                decisions[output.decision.decision_id] = output.decision
                rollout_steps.append(rollout_step)
            if self.buffer is not None:
                add_step = getattr(self.buffer, "add_step", None)
                if rollout_step is not None and callable(add_step):
                    add_step(rollout_step, output.decision)
                else:
                    self.buffer.add(transition)

            self._episode_length += 1
            self._accumulate_reward(reward)
            self._episode_success = self._episode_success or transition.success
            self._obs = next_obs
            if transition.episode_done:
                episodes.append(self._finish_episode(transition))

        rollout = None
        if rollout_steps:
            rollout = Rollout(
                steps=tuple(rollout_steps),
                decisions=tuple(decisions.values()),
            )
        return CollectResult(
            transitions=tuple(transitions),
            episodes=tuple(episodes),
            rollout=rollout,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.executor.close()
        finally:
            self.runner.close()
            self._closed = True

    def policy_updated(self) -> None:
        self._ensure_open()
        self.executor.policy_updated()
