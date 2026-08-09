"""State-only Gymnasium environments exposed through ILStudio contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from benchmark.base import MetaAction, MetaEnv, MetaObs
from benchmark.utils import normalize_step_result


def create_env(config):
    return GymEnv(config)


def _namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _namespace_to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {key: _namespace_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    return value


class GymEnv(MetaEnv):
    """Adapt a Gymnasium state environment to ``MetaEnv``.

    The adapter intentionally does not render an image. Observations are
    flattened with Gymnasium's space utilities and stored in ``MetaObs.state``;
    ``MetaObs.image`` remains ``None``. Discrete and Box action spaces are
    supported so the same benchmark can validate value- and policy-based RL.
    """

    def __init__(self, config):
        self.config = config
        self.task = getattr(config, "task", None)
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("GymEnv config requires a non-empty task")

        self.ctrl_space = getattr(config, "ctrl_space", "gym")
        self.raw_lang = getattr(config, "raw_lang", self.task)
        self.clip_action = bool(getattr(config, "clip_action", True))
        self.success_on_time_limit = bool(
            getattr(config, "success_on_time_limit", False)
        )
        self.success_return_threshold = getattr(
            config, "success_return_threshold", None
        )
        self.seed = getattr(config, "seed", None)
        if self.seed is not None:
            self.seed = int(self.seed)

        make_kwargs = _namespace_to_dict(getattr(config, "make_kwargs", {}))
        max_timesteps = getattr(config, "max_timesteps", None)
        if max_timesteps is not None:
            make_kwargs.setdefault("max_episode_steps", int(max_timesteps))
        render_mode = getattr(config, "render_mode", None)
        if render_mode is not None:
            make_kwargs.setdefault("render_mode", render_mode)

        env = gym.make(self.task, **make_kwargs)
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.state_dim = int(spaces.flatdim(self.observation_space))

        if isinstance(self.action_space, spaces.Discrete):
            self.ctrl_type = "discrete"
            self.action_dim = 1
            self.num_actions = int(self.action_space.n)
        elif isinstance(self.action_space, spaces.Box):
            self.ctrl_type = "continuous"
            self.action_dim = int(spaces.flatdim(self.action_space))
            self.num_actions = None
        else:
            env.close()
            raise TypeError(
                "GymEnv currently supports Discrete and Box action spaces, "
                f"got {type(self.action_space).__name__}"
            )

        self._has_seeded = False
        self._timestep = 0
        self._episode_return = 0.0

    def obs2meta(self, raw_obs):
        state = spaces.flatten(self.observation_space, raw_obs)
        state = np.asarray(state, dtype=np.float32)
        return MetaObs(
            state=state,
            image=None,
            raw_lang=self.raw_lang,
            timestep=self._timestep,
        )

    def meta2act(self, action: MetaAction):
        if not isinstance(action, MetaAction):
            raise TypeError("GymEnv action must be a MetaAction")
        if action.ctrl_space != self.ctrl_space:
            raise ValueError(
                f"MetaAction ctrl_space {action.ctrl_space!r} does not match "
                f"GymEnv ctrl_space {self.ctrl_space!r}"
            )
        if action.ctrl_type != self.ctrl_type:
            raise ValueError(
                f"MetaAction ctrl_type {action.ctrl_type!r} does not match "
                f"GymEnv ctrl_type {self.ctrl_type!r}"
            )
        if action.action is None:
            raise ValueError("MetaAction.action cannot be None")

        value = np.asarray(action.action)
        if isinstance(self.action_space, spaces.Discrete):
            flat = value.reshape(-1)
            if flat.size != 1:
                raise ValueError("a Discrete Gym action must contain one value")
            scalar = flat.item()
            discrete = int(scalar)
            if float(discrete) != float(scalar):
                raise ValueError("a Discrete Gym action must be integer-valued")
            if not self.action_space.contains(discrete):
                raise ValueError(f"action {discrete} is outside {self.action_space}")
            return discrete

        expected = self.action_dim
        flat = value.astype(self.action_space.dtype, copy=False).reshape(-1)
        if flat.size != expected:
            raise ValueError(
                f"a Box Gym action needs {expected} values, got {flat.size}"
            )
        native = flat.reshape(self.action_space.shape)
        if self.clip_action:
            native = np.clip(native, self.action_space.low, self.action_space.high)
        if not self.action_space.contains(native):
            raise ValueError(f"action is outside {self.action_space}")
        return native

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> MetaObs:
        if seed is None and not self._has_seeded:
            seed = self.seed
        kwargs = {"seed": seed}
        if options is not None:
            kwargs["options"] = options
        raw_obs, _ = self.env.reset(**kwargs)
        self._has_seeded = True
        self._timestep = 0
        self._episode_return = 0.0
        self.prev_obs = self.obs2meta(raw_obs)
        return self.prev_obs

    def step(self, action: MetaAction):
        native_action = self.meta2act(action)
        raw_obs, reward, terminated, truncated, info = normalize_step_result(
            self.env.step(native_action)
        )
        self._timestep += 1
        self._episode_return += float(reward)
        self.prev_obs = self.obs2meta(raw_obs)

        info = dict(info)
        success = bool(info.get("success", info.get("is_success", False)))
        if terminated or truncated:
            if self.success_on_time_limit and truncated and not terminated:
                success = True
            if self.success_return_threshold is not None:
                success = self._episode_return >= float(
                    self.success_return_threshold
                )
            info["episode_return"] = self._episode_return
        info["success"] = success
        info["task"] = self.task
        return self.prev_obs, reward, terminated, truncated, info


__all__ = ["GymEnv", "create_env"]
