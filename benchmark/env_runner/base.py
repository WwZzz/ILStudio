"""Base interface for environment execution around ``MetaEnv`` instances."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from benchmark.base import MetaAction, MetaObs


EnvStep = Tuple[MetaObs, Any, bool, bool, Dict[str, Any]]


class BaseEnvRunner(ABC):
    """Own environment lifecycle, but not policy, reward, or buffer logic."""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of environments controlled by this runner."""

    @property
    @abstractmethod
    def needs_reset(self) -> bool:
        """Whether ``reset`` must be called before another ``step``."""

    @abstractmethod
    def reset(self) -> MetaObs:
        """Reset the environment and return an ILStudio ``MetaObs``."""

    @abstractmethod
    def step(self, action: MetaAction) -> EnvStep:
        """Execute one ``MetaAction`` using the Gymnasium five-value API."""

    @abstractmethod
    def close(self) -> None:
        """Release environment resources. Implementations must be idempotent."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
