"""Policy execution boundaries used by RL collectors."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional

from benchmark.base import MetaObs
from rl.base import PolicyOutput


class BasePolicyExecutor(ABC):
    """Execute policies for collection without assuming a transport mechanism."""

    @abstractmethod
    def select_action(
        self,
        obs: MetaObs,
        *,
        deterministic: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PolicyOutput:
        """Return one environment-step action."""

    @abstractmethod
    def reset(self) -> None:
        """Clear episode-local execution state such as action chunks."""

    @abstractmethod
    def close(self) -> None:
        """Release executor resources. Implementations must be idempotent."""

    def policy_updated(self) -> None:
        """Discard execution state that was produced by the previous policy."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
