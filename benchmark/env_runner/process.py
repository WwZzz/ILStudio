"""Spawned multi-process execution for independent ILStudio environments."""

import copy
import multiprocessing as mp
import random
import traceback
from collections.abc import Mapping, Sequence

from benchmark.base import MetaAction, MetaEnv

from .base import BaseEnvRunner, MetaEnvSpec
from .sync import SyncEnvRunner


def _worker_config(config, env_index, seed_stride, base_seed):
    config = copy.deepcopy(config)
    seed = base_seed
    if seed is None:
        seed = (
            config.get("seed")
            if isinstance(config, Mapping)
            else getattr(config, "seed", None)
        )
    if seed is not None:
        worker_seed = int(seed) + env_index * seed_stride
        if isinstance(config, Mapping):
            config["seed"] = worker_seed
        else:
            setattr(config, "seed", worker_seed)
    return config


def _worker_runner(factory, config, env_index, seed_stride, base_seed):
    config = _worker_config(config, env_index, seed_stride, base_seed)
    env = factory(config)
    if not isinstance(env, MetaEnv):
        raise TypeError("environment factory must return an ILStudio MetaEnv")
    return SyncEnvRunner(env)


def _worker_main(
    connection, env_factory, config, env_index, seed_stride, base_seed
):
    runner = None
    try:
        runner = _worker_runner(
            env_factory, config, env_index, seed_stride, base_seed
        )
        connection.send((True, "ready"))
        while True:
            operation, value = connection.recv()
            if operation == "close":
                connection.send((True, None))
                break
            try:
                if operation == "reconfigure":
                    factory, next_config, next_seed = value
                    runner.close()
                    runner = _worker_runner(
                        factory,
                        next_config,
                        env_index,
                        seed_stride,
                        next_seed,
                    )
                    connection.send((True, "ready"))
                    continue
                method = getattr(runner, operation)
                result = method(value) if value is not None else method()
                connection.send((True, result))
            except BaseException as exc:
                connection.send(
                    (
                        False,
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    )
                )
    except BaseException as exc:
        connection.send(
            (
                False,
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )
        )
    finally:
        if runner is not None:
            runner.close()
        connection.close()


class _ProcessWorker:
    def __init__(
        self,
        context,
        env_factory,
        config,
        env_index,
        *,
        seed_stride,
        seed,
        timeout,
    ):
        parent, child = context.Pipe()
        self.connection = parent
        self.env_index = env_index
        self.timeout = timeout
        self.process = context.Process(
            target=_worker_main,
            args=(child, env_factory, config, env_index, seed_stride, seed),
            daemon=True,
            name=f"ilstudio-env-{env_index}",
        )
        self.process.start()
        child.close()

    def request(self, operation, value=None):
        if not self.process.is_alive():
            raise RuntimeError(f"environment worker {self.env_index} is not alive")
        self.connection.send((operation, value))

    def result(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f"environment worker {self.env_index} timed out")
        ok, value = self.connection.recv()
        if not ok:
            raise RuntimeError(
                f"environment worker {self.env_index} failed with "
                f"{value['type']}: {value['message']}\n{value['traceback']}"
            )
        return value

    def close(self):
        if self.process.is_alive():
            try:
                self.request("close")
                self.result()
            except (BrokenPipeError, EOFError, OSError, RuntimeError, TimeoutError):
                pass
        self.process.join(timeout=self.timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=self.timeout)
        self.connection.close()


class ProcessEnvRunner(BaseEnvRunner):
    """Run independent MetaEnv instances in spawned worker processes.

    A prototype environment is supplied by ILStudio's normal config builder.
    Workers reconstruct the same adapter from its class and config, so CUDA
    policy objects are never copied into environment processes.
    """

    def __init__(
        self,
        env: MetaEnv = None,
        *,
        env_spec: MetaEnvSpec = None,
        num_envs: int = 2,
        start_method: str = "spawn",
        seed_stride: int = 1000000,
        seed=None,
        timeout: float = 120.0,
    ) -> None:
        if (env is None) == (env_spec is None):
            raise ValueError("provide exactly one of env or env_spec")
        if env is not None and not isinstance(env, MetaEnv):
            raise TypeError("env must be an ILStudio MetaEnv")
        if env_spec is not None and not isinstance(env_spec, MetaEnvSpec):
            raise TypeError("env_spec must be MetaEnvSpec")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 1:
            raise ValueError("ProcessEnvRunner requires num_envs greater than one")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method {start_method!r}")
        if isinstance(seed_stride, bool) or not isinstance(seed_stride, int) or seed_stride < 0:
            raise ValueError("seed_stride must be a non-negative integer")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if env_spec is None:
            config = getattr(env, "config", None)
            if config is None:
                raise TypeError("parallel environments require the MetaEnv config")
            env_spec = MetaEnvSpec(type(env), config)
            SyncEnvRunner(env).close()

        self._num_envs = num_envs
        self._closed = False
        self._needs_reset = [True] * num_envs
        self._last_obs = [None] * num_envs
        self._context = mp.get_context(start_method)
        self._workers = []
        try:
            self._workers = [
                _ProcessWorker(
                    self._context,
                    env_spec.factory,
                    env_spec.config,
                    index,
                    seed_stride=seed_stride,
                    seed=seed,
                    timeout=float(timeout),
                )
                for index in range(num_envs)
            ]
            for worker in self._workers:
                if worker.result() != "ready":
                    raise RuntimeError("environment worker returned an invalid handshake")
        except BaseException:
            self.close()
            raise

    def _reconfigure(self, env_spec, *, seed=None):
        self._ensure_open()
        if not isinstance(env_spec, MetaEnvSpec):
            raise TypeError("env_spec must be MetaEnvSpec")
        for worker in self._workers:
            worker.request(
                "reconfigure", (env_spec.factory, env_spec.config, seed)
            )
        for worker in self._workers:
            if worker.result() != "ready":
                raise RuntimeError("environment worker returned an invalid handshake")
        self._needs_reset = [True] * self.num_envs
        self._last_obs = [None] * self.num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def needs_reset(self) -> bool:
        return any(self._needs_reset)

    @property
    def needs_reset_mask(self):
        return tuple(self._needs_reset)

    @property
    def supports_partial_reset(self) -> bool:
        return True

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("environment runner is closed")

    def _indices(self, env_indices):
        if env_indices is None:
            env_indices = tuple(range(self.num_envs))
        else:
            env_indices = tuple(env_indices)
        if not env_indices:
            raise ValueError("env_indices cannot be empty")
        if len(set(env_indices)) != len(env_indices):
            raise ValueError("env_indices must be unique")
        for index in env_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("environment indices must be integers")
            if not 0 <= index < self.num_envs:
                raise IndexError("environment index is out of range")
        return env_indices

    def reset(self, env_indices=None):
        self._ensure_open()
        env_indices = self._indices(env_indices)
        for index in env_indices:
            self._workers[index].request("reset")
        observations = []
        for index in env_indices:
            obs = self._workers[index].result()
            self._needs_reset[index] = False
            self._last_obs[index] = obs
            observations.append(obs)
        return tuple(observations)

    def step(self, actions, *, env_indices=None):
        self._ensure_open()
        actions = tuple(actions)
        env_indices = self._indices(env_indices)
        if len(actions) != len(env_indices):
            raise ValueError("actions and env_indices must have equal length")
        if not all(isinstance(action, MetaAction) for action in actions):
            raise TypeError("actions must contain MetaAction values")
        for index in env_indices:
            if self._needs_reset[index]:
                raise RuntimeError(f"environment {index} must be reset before step")
        for index, action in zip(env_indices, actions):
            self._workers[index].request("step", action)
        results = []
        for index in env_indices:
            result = self._workers[index].result()
            obs, _, terminated, truncated, _ = result
            self._needs_reset[index] = bool(terminated or truncated)
            self._last_obs[index] = obs
            results.append(result)
        return tuple(results)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for worker in self._workers:
                worker.close()
        finally:
            self._closed = True
            self._needs_reset = [True] * self._num_envs


class GroupedProcessEnvRunner(ProcessEnvRunner):
    """Schedule one environment/trial prompt across a replica group.

    This preserves the normal ILStudio environment adapters and process runner.
    The only added policy is which configured environment and episode index the
    whole replica group receives before its next full reset.
    """

    def __init__(
        self,
        envs: Sequence,
        *,
        num_envs: int = 8,
        episodes_per_env: int = 50,
        episode_index_key: str = "init_state_index",
        shuffle: bool = True,
        schedule_seed: int = 0,
        **kwargs,
    ):
        envs = tuple(envs)
        if not envs or not all(isinstance(item, MetaEnvSpec) for item in envs):
            raise TypeError("envs must be a non-empty sequence of MetaEnvSpec")
        if (
            isinstance(episodes_per_env, bool)
            or not isinstance(episodes_per_env, int)
            or episodes_per_env <= 0
        ):
            raise ValueError("episodes_per_env must be a positive integer")
        if not isinstance(episode_index_key, str) or not episode_index_key:
            raise TypeError("episode_index_key must be a non-empty string")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be bool")
        if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
            raise TypeError("schedule_seed must be an integer")

        self._env_specs = envs
        self.episodes_per_env = episodes_per_env
        self.episode_index_key = episode_index_key
        self.shuffle = shuffle
        self._schedule_rng = random.Random(schedule_seed)
        self._schedule = []
        self._schedule_offset = 0
        self.schedule_epoch = -1
        self.current_env_config_index = None
        self.current_episode_index = None
        super().__init__(
            env_spec=envs[0], num_envs=num_envs, **kwargs
        )

    def _start_schedule_epoch(self):
        self._schedule = [
            (env_index, episode_index)
            for env_index in range(len(self._env_specs))
            for episode_index in range(self.episodes_per_env)
        ]
        if self.shuffle:
            self._schedule_rng.shuffle(self._schedule)
        self._schedule_offset = 0
        self.schedule_epoch += 1

    def _next_prompt(self):
        if self._schedule_offset >= len(self._schedule):
            self._start_schedule_epoch()
        item = self._schedule[self._schedule_offset]
        self._schedule_offset += 1
        return item

    def _scheduled_spec(self, env_index, episode_index):
        source = self._env_specs[env_index]
        config = copy.deepcopy(source.config)
        if isinstance(config, Mapping):
            config[self.episode_index_key] = episode_index
        else:
            setattr(config, self.episode_index_key, episode_index)
        nested_args = (
            config.get("args")
            if isinstance(config, Mapping)
            else getattr(config, "args", None)
        )
        if isinstance(nested_args, Mapping):
            nested_args[self.episode_index_key] = episode_index
        elif nested_args is not None:
            setattr(nested_args, self.episode_index_key, episode_index)
        return MetaEnvSpec(source.factory, config)

    def reset(self, env_indices=None):
        requested = self._indices(env_indices)
        if all(self._needs_reset):
            expected = tuple(range(self.num_envs))
            if requested != expected:
                raise RuntimeError(
                    "a completed replica group must be reset as a whole"
                )
            env_config_index, episode_index = self._next_prompt()
            self._reconfigure(
                self._scheduled_spec(env_config_index, episode_index)
            )
            self.current_env_config_index = env_config_index
            self.current_episode_index = episode_index
        return super().reset(requested)

    def step(self, actions, *, env_indices=None):
        results = super().step(actions, env_indices=env_indices)
        annotated = []
        for obs, reward, terminated, truncated, info in results:
            info = dict(info)
            info["env_config_index"] = self.current_env_config_index
            info["episode_index"] = self.current_episode_index
            info["schedule_epoch"] = self.schedule_epoch
            annotated.append((obs, reward, terminated, truncated, info))
        return tuple(annotated)
