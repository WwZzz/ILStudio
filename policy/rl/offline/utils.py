"""
Common utilities for offline RL algorithms.

This module provides shared functions for all offline RL algorithms in ILStudio.
"""

from typing import Optional, Tuple, Any
import numpy as np
from loguru import logger


def load_d4rl_to_replay_buffer(
    env_name: str,
    capacity: Optional[int] = None,
    device: str = 'cuda',
    storage_device: str = 'cpu',
    show_progress: bool = True,
) -> Tuple[Any, 'ILReplayBuffer']:
    """
    Load D4RL dataset into ILStudio's ILReplayBuffer.
    
    This is the standard way to load offline RL datasets in ILStudio.
    All offline RL algorithms (IQL, CQL, BCQ, etc.) should use this function.
    
    Args:
        env_name: D4RL environment name (e.g., 'halfcheetah-medium-v2')
        capacity: Buffer capacity (None = use dataset size)
        device: Device for sampling
        storage_device: Device for storage (use 'cpu' to save GPU memory)
        show_progress: Show loading progress bar
        
    Returns:
        Tuple of (gym environment, ILReplayBuffer with loaded data)
        
    Example:
        >>> from policy.rl.offline.utils import load_d4rl_to_replay_buffer
        >>> env, replay_buffer = load_d4rl_to_replay_buffer('halfcheetah-medium-v2')
        >>> print(f"Loaded {replay_buffer.size} transitions")
    """
    # IMPORTANT:
    # D4RL registers environments into Gym on import-time side effects.
    # So we must import d4rl BEFORE calling gym.make(env_name).
    try:
        import d4rl  # noqa: F401
        import gym
    except ImportError:
        raise ImportError(
            "d4rl and gym are required. Install with: pip install d4rl gym"
        )
    
    from policy.rl.replay_buffer import ILReplayBuffer
    from benchmark.base import MetaObs, MetaAction
    
    # Load D4RL dataset
    try:
        env = gym.make(env_name)
    except Exception as e:
        # Common failure: Mujoco envs are not registered because Mujoco deps are missing.
        # D4RL prints: "Mujoco-based envs failed to import".
        msg = (
            f"Failed to gym.make('{env_name}'). This usually means the D4RL env was not registered.\n"
            f"- If you're using Mujoco tasks (halfcheetah/hopper/walker2d/ant), please install Mujoco deps "
            f"so D4RL can register those envs.\n"
            f"- Alternatively, try a Bullet D4RL env name (the error hint often suggests something like "
            f"'bullet-<env>').\n"
            f"Original error: {repr(e)}"
        )
        raise RuntimeError(msg) from e
    dataset = d4rl.qlearning_dataset(env)
    
    dataset_size = len(dataset['observations'])
    if capacity is None:
        capacity = dataset_size
    
    obs_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]
    
    logger.info(f"Loading D4RL dataset '{env_name}' into ILReplayBuffer")
    logger.info(f"  Dataset size: {dataset_size:,}")
    logger.info(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create replay buffer
    replay_buffer = ILReplayBuffer(
        capacity=capacity,
        chunk_size=1,
        ctrl_space='joint',
        ctrl_type='abs',
        device=device,
        storage_device=storage_device,
    )
    
    # Add transitions to buffer
    iterator = range(dataset_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Loading to ReplayBuffer")
    
    for i in iterator:
        obs = MetaObs(state=dataset['observations'][i].astype(np.float32))
        next_obs = MetaObs(state=dataset['next_observations'][i].astype(np.float32))
        action = MetaAction(
            action=dataset['actions'][i].astype(np.float32),
            ctrl_space='joint',
            ctrl_type='abs'
        )
        reward = float(dataset['rewards'][i])
        done = bool(dataset['terminals'][i])
        
        # D4RL uses 'timeouts' for truncation
        truncated = bool(dataset.get('timeouts', np.zeros(dataset_size))[i])
        
        replay_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            truncated=truncated,
        )
    
    logger.info(f"ReplayBuffer loaded: {replay_buffer.size:,} transitions")
    
    return env, replay_buffer

