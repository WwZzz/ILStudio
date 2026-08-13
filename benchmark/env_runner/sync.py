"""Single-environment runner for existing ILStudio ``MetaEnv`` adapters."""

from typing import Mapping

from benchmark.base import MetaAction, MetaEnv, MetaObs, dict2meta
from benchmark.utils import normalize_step_result

from .base import BaseEnvRunner, EnvStep


class SyncEnvRunner(BaseEnvRunner):
    """Synchronous, non-vectorized execution around one ``MetaEnv``.

    The runner deliberately has no policy dependency.  A collector coordinates
    it with a policy executor later.  It also performs no auto-reset so episode
    boundaries cannot be hidden from an RL buffer.
    """

    def __init__(self, env: MetaEnv, *, max_episode_steps=None):
        if not isinstance(env, MetaEnv):
            raise TypeError("env must be an ILStudio MetaEnv")
        if max_episode_steps is None:
            config = getattr(env, "config", None)
            max_episode_steps = getattr(config, "max_timesteps", None)
        if max_episode_steps is not None and (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer or None")
        self.env = env
        self.max_episode_steps = max_episode_steps
        self._closed = False
        self._needs_reset = True
        self._has_reset = False
        self._episode_steps = 0
        self._episode_index = -1
        self._last_obs = None

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def needs_reset(self) -> bool:
        return self._needs_reset

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def last_obs(self):
        return self._last_obs

    @property
    def supports_partial_reset(self) -> bool:
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment runner is closed")

    @staticmethod
    def _ensure_meta_obs(obs) -> MetaObs:
        if isinstance(obs, MetaObs):
            return obs
        if isinstance(obs, Mapping):
            return dict2meta(dict(obs), mtype="obs")
        raise TypeError(
            "MetaEnv reset/step must expose MetaObs or a MetaObs-compatible dict"
        )

    def reset(self) -> MetaObs:
        self._ensure_open()
        obs = self._ensure_meta_obs(self.env.reset())
        self._has_reset = True
        self._needs_reset = False
        self._episode_steps = 0
        self._episode_index += 1
        self._last_obs = obs
        return obs

    def step(self, action: MetaAction) -> EnvStep:
        self._ensure_open()
        if not self._has_reset:
            raise RuntimeError("call reset() before step()")
        if self._needs_reset:
            raise RuntimeError("episode has ended; call reset() before step()")
        if not isinstance(action, MetaAction):
            raise TypeError("action must be an ILStudio MetaAction")

        obs, reward, terminated, truncated, info = normalize_step_result(
            self.env.step(action)
        )
        obs = self._ensure_meta_obs(obs)

        self._episode_steps += 1
        if (
            self.max_episode_steps is not None
            and self._episode_steps >= self.max_episode_steps
            and not terminated
            and not truncated
        ):
            truncated = True
            info = dict(info)
            info["terminated"] = False
            info["truncated"] = True
            info["TimeLimit.truncated"] = True
        self._needs_reset = terminated or truncated
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        try:
            try:
                self.env.close()
            finally:
                # Some benchmark adapters keep close() reusable for eval but
                # expose force_close() for the lifecycle owner.  SyncEnvRunner
                # owns its environment, so finalize it before interpreter exit.
                force_close = getattr(self.env, "force_close", None)
                if callable(force_close):
                    force_close()
        finally:
            self._closed = True
            self._needs_reset = True
