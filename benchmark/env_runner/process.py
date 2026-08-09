"""Spawned multi-process execution for independent ILStudio environments."""

import copy
import multiprocessing as mp
import traceback
from collections.abc import Mapping, Sequence

from benchmark.base import MetaAction, MetaEnv

from .base import BaseEnvRunner
from .sync import SyncEnvRunner


def _worker_main(connection, env_class, config, env_index, seed_stride):
    runner = None
    try:
        config = copy.deepcopy(config)
        seed = (
            config.get("seed") if isinstance(config, Mapping) else getattr(config, "seed", None)
        )
        if seed is not None:
            worker_seed = int(seed) + env_index * seed_stride
            if isinstance(config, Mapping):
                config["seed"] = worker_seed
            else:
                setattr(config, "seed", worker_seed)
        runner = SyncEnvRunner(env_class(config))
        connection.send((True, "ready"))
        while True:
            operation, value = connection.recv()
            if operation == "close":
                connection.send((True, None))
                break
            try:
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
        env_class,
        config,
        env_index,
        *,
        seed_stride,
        timeout,
    ):
        parent, child = context.Pipe()
        self.connection = parent
        self.env_index = env_index
        self.timeout = timeout
        self.process = context.Process(
            target=_worker_main,
            args=(child, env_class, config, env_index, seed_stride),
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
        env: MetaEnv,
        *,
        num_envs: int = 2,
        start_method: str = "spawn",
        seed_stride: int = 1000000,
        timeout: float = 120.0,
    ) -> None:
        if not isinstance(env, MetaEnv):
            raise TypeError("env must be an ILStudio MetaEnv")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 1:
            raise ValueError("ProcessEnvRunner requires num_envs greater than one")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method {start_method!r}")
        if isinstance(seed_stride, bool) or not isinstance(seed_stride, int) or seed_stride <= 0:
            raise ValueError("seed_stride must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        config = getattr(env, "config", None)
        if config is None:
            raise TypeError("parallel environments require the MetaEnv config")
        env_class = type(env)
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
                    env_class,
                    config,
                    index,
                    seed_stride=seed_stride,
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
