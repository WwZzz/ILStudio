"""Reserved reward keys and environment reward conversion."""

from typing import Any, Dict


ENV_REWARD_KEY = "env/raw"
TOTAL_REWARD_KEY = "train/total"


def wrap_env_reward(reward: Any) -> Dict[str, Any]:
    """Wrap a raw Gymnasium reward without transforming or overwriting it."""

    if reward is None:
        return {}
    return {ENV_REWARD_KEY: reward}
