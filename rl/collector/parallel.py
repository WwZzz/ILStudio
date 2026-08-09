"""Synchronous batched collection over independent parallel environments."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Optional

from benchmark.env_runner import BaseEnvRunner
from rl.base import MetaTransition, Rollout, RolloutStep
from rl.buffer import BaseBuffer
from rl.executor import BasePolicyExecutor
from rl.reward import RewardComposer, wrap_env_reward

from .base import (
    BaseCollector,
    CollectResult,
    EpisodeSummary,
    annotate_episode_timestep,
    apply_episode_time_limit,
    apply_episode_semantics,
    validate_episode_semantics,
    validate_max_episode_steps,
)
from .sync import _validate_target


class ParallelCollector(BaseCollector):
    """Step several envs together and batch policy/reward computation."""

    def __init__(
        self,
        *,
        runner: BaseEnvRunner,
        executor: BasePolicyExecutor,
        buffer: Optional[BaseBuffer] = None,
        reward_composer: Optional[RewardComposer] = None,
        terminate_on_success: bool = False,
        bootstrap_on_truncation: bool = True,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        if not isinstance(runner, BaseEnvRunner):
            raise TypeError("runner must inherit BaseEnvRunner")
        if runner.num_envs <= 1:
            raise ValueError("ParallelCollector requires a multi-environment runner")
        if not isinstance(executor, BasePolicyExecutor):
            raise TypeError("executor must inherit BasePolicyExecutor")
        if getattr(executor, "num_envs", None) != runner.num_envs:
            raise ValueError("runner and executor num_envs must match")
        if not callable(getattr(executor, "select_actions", None)):
            raise TypeError("parallel executor must provide select_actions()")
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
        self.max_episode_steps = validate_max_episode_steps(max_episode_steps)
        self._observations = [None] * runner.num_envs
        self._episode_lengths = [0] * runner.num_envs
        self._episode_rewards = [{} for _ in range(runner.num_envs)]
        self._episode_successes = [False] * runner.num_envs
        self._episode_index = 0
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("collector is closed")

    def _active_indices(self, limit):
        active = [
            index
            for index, obs in enumerate(self._observations)
            if obs is not None
        ]
        inactive = [
            index
            for index, obs in enumerate(self._observations)
            if obs is None
        ]
        return tuple((active + inactive)[:limit])

    def _ensure_episodes(self, env_indices):
        reset_indices = tuple(
            index for index in env_indices if self._observations[index] is None
        )
        if not reset_indices:
            return
        self.executor.reset(reset_indices)
        observations = self.runner.reset(reset_indices)
        if len(observations) != len(reset_indices):
            raise ValueError("parallel reset result size does not match env_indices")
        for index, obs in zip(reset_indices, observations):
            self._observations[index] = annotate_episode_timestep(
                obs, 0
            )
            self._episode_lengths[index] = 0
            self._episode_rewards[index] = {}
            self._episode_successes[index] = False

    def _accumulate_reward(self, env_index, reward):
        total = self._episode_rewards[env_index]
        for key, value in reward.items():
            total[key] = total[key] + value if key in total else value

    def _finish_episode(self, env_index, transition):
        summary = EpisodeSummary(
            index=self._episode_index,
            length=self._episode_lengths[env_index],
            reward=self._episode_rewards[env_index],
            terminated=transition.terminated,
            truncated=transition.truncated,
            success=self._episode_successes[env_index],
            info=transition.info,
        )
        self._episode_index += 1
        self._observations[env_index] = None
        self._episode_lengths[env_index] = 0
        self._episode_rewards[env_index] = {}
        self._episode_successes[env_index] = False
        self.executor.reset((env_index,))
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
        while True:
            if num_steps is not None:
                remaining = num_steps - len(transitions)
            else:
                remaining = num_episodes - len(episodes)
            if remaining <= 0:
                break
            env_indices = self._active_indices(min(self.runner.num_envs, remaining))
            self._ensure_episodes(env_indices)
            observations = tuple(self._observations[index] for index in env_indices)
            outputs = tuple(
                self.executor.select_actions(
                    observations,
                    env_indices=env_indices,
                    deterministic=deterministic,
                    context=context,
                )
            )
            step_results = tuple(
                self.runner.step(
                    tuple(output.action for output in outputs),
                    env_indices=env_indices,
                )
            )
            if len(outputs) != len(env_indices) or len(step_results) != len(env_indices):
                raise ValueError("parallel step results do not align with env_indices")

            raw_transitions = []
            for env_index, obs, output, step_result in zip(
                env_indices, observations, outputs, step_results
            ):
                next_obs, env_reward, terminated, truncated, info = step_result
                next_obs = annotate_episode_timestep(
                    next_obs,
                    self._episode_lengths[env_index] + 1,
                )
                info = dict(info)
                existing_index = info.get("env_index", env_index)
                if existing_index != env_index:
                    raise ValueError("environment returned a conflicting env_index")
                info["env_index"] = env_index
                transition = apply_episode_time_limit(
                    MetaTransition(
                        obs=obs,
                        action=output.action,
                        next_obs=next_obs,
                        reward=wrap_env_reward(env_reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                        policy_info=output.policy_info,
                    ),
                    episode_step=self._episode_lengths[env_index] + 1,
                    max_episode_steps=self.max_episode_steps,
                )
                raw_transitions.append(
                    apply_episode_semantics(
                        transition,
                        terminate_on_success=self.terminate_on_success,
                        bootstrap_on_truncation=self.bootstrap_on_truncation,
                    )
                )
            rewards = self.reward_composer.compute_batch(
                tuple(raw_transitions), context=context
            )

            for env_index, output, raw_transition, reward in zip(
                env_indices, outputs, raw_transitions, rewards
            ):
                transition = replace(raw_transition, reward=reward)
                transitions.append(transition)
                rollout_step = None
                if output.decision is not None:
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

                self._episode_lengths[env_index] += 1
                self._accumulate_reward(env_index, reward)
                self._episode_successes[env_index] = (
                    self._episode_successes[env_index] or transition.success
                )
                self._observations[env_index] = transition.next_obs
                if transition.episode_done:
                    episodes.append(self._finish_episode(env_index, transition))

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
