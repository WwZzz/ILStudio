"""Task-success rewards derived from the canonical transition metadata."""

from typing import Any, Dict

from rl.base import MetaTransition

from .base import BaseReward, RewardContext, RewardDict


class SuccessReward(BaseReward):
    """Emit a task-success component, optionally only on the first success.

    ``first_only=True`` is useful for environments whose success flag remains
    high after completion: the whole episode receives one bonus instead of a
    duration-dependent reward.  Episode state resets only at Gymnasium's
    terminated/truncated boundary and is checkpointed for exact continuation.
    """

    STATE_VERSION = 2

    def __init__(
        self,
        namespace: str = "task",
        *,
        first_only: bool = True,
    ) -> None:
        super().__init__(namespace)
        if not isinstance(first_only, bool):
            raise TypeError("first_only must be bool")
        self.first_only = first_only
        self._episode_has_succeeded = {}

    def compute_step(
        self,
        transition: MetaTransition,
        *,
        context: RewardContext = None,
    ) -> RewardDict:
        del context
        env_index = transition.info.get("env_index", 0)
        if isinstance(env_index, bool) or not isinstance(env_index, int) or env_index < 0:
            raise ValueError("transition env_index must be a non-negative integer")
        succeeded = transition.success
        already_succeeded = self._episode_has_succeeded.get(env_index, False)
        reward = float(
            succeeded
            and (not self.first_only or not already_succeeded)
        )
        if transition.episode_done:
            self._episode_has_succeeded.pop(env_index, None)
        else:
            self._episode_has_succeeded[env_index] = already_succeeded or succeeded
        return {"success": reward}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "first_only": self.first_only,
            "episode_has_succeeded": dict(self._episode_has_succeeded),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("success reward state must be a dict")
        version = state.get("version")
        if version not in {1, self.STATE_VERSION}:
            raise ValueError("unsupported success reward state version")
        if state.get("first_only") is not self.first_only:
            raise ValueError("success reward first_only does not match checkpoint")
        seen = state.get("episode_has_succeeded")
        if version == 1:
            if not isinstance(seen, bool):
                raise TypeError("episode_has_succeeded must be bool")
            self._episode_has_succeeded = {0: True} if seen else {}
            return
        if not isinstance(seen, dict):
            raise TypeError("episode_has_succeeded must be a dict")
        if not all(
            isinstance(key, int)
            and not isinstance(key, bool)
            and key >= 0
            and isinstance(value, bool)
            for key, value in seen.items()
        ):
            raise TypeError("episode success state must map env indices to bool")
        self._episode_has_succeeded = dict(seen)
