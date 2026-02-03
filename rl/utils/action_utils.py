"""
Action utility helpers for RL algorithms.

These helpers keep action post-processing outside MetaEnv, so algorithms can
apply different refinement strategies when needed.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch


def _get_action_space(env) -> Optional[object]:
    """Best-effort extraction of an action_space from env or wrappers."""
    if env is None:
        return None
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    if hasattr(env, "envs") and env.envs:
        first_env = env.envs[0]
        if hasattr(first_env, "action_space"):
            return first_env.action_space
        if hasattr(first_env, "env") and hasattr(first_env.env, "action_space"):
            return first_env.env.action_space
    return None


def _get_action_bounds(env) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    action_space = _get_action_space(env)
    if action_space is None:
        return None, None
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return None, None
    return np.asarray(low), np.asarray(high)


def clip_action_to_space(env, action):
    """Clip action to env action space bounds if available."""
    low, high = _get_action_bounds(env)
    if low is None or high is None:
        return action
    if torch.is_tensor(action):
        low_t = torch.as_tensor(low, device=action.device, dtype=action.dtype)
        high_t = torch.as_tensor(high, device=action.device, dtype=action.dtype)
        return torch.clamp(action, low_t, high_t)
    return np.clip(action, low, high)


def ensure_action(
    env,
    action,
    refine_fn: Optional[Callable[[object, object], object]] = None,
    apply_tanh: bool = True,
):
    """
    Ensure actions are valid for the environment.

    - Applies tanh to bound outputs to [-1, 1] (as in TD3 policies).
    - Applies an optional refine function for env-specific processing.
    - Clips to env action space bounds if available.

    Args:
        env: Environment (or wrapper) providing action_space.
        action: Tensor or numpy array action.
        refine_fn: Optional callable (env, action) -> action for custom refinement.
        apply_tanh: Whether to apply tanh to the action.
    """
    if apply_tanh:
        action = torch.tanh(action) if torch.is_tensor(action) else np.tanh(action)
    if refine_fn is not None:
        action = refine_fn(env, action)
    return clip_action_to_space(env, action)

