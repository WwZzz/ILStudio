"""
Rollout Replay Buffer for parallel environment data collection.

This module provides a specialized replay buffer that handles data collection
from multiple parallel environments, tracking episode IDs and timestamps
for proper episode segmentation.
"""

import numpy as np
from typing import Optional, Dict, List, Any

from benchmark.base import MetaAction
from .buffer import ILReplayBuffer


class RolloutReplayBuffer(ILReplayBuffer):
    """
    Extended replay buffer for rollout data collection.
    
    Handles parallel environments by tracking per-environment episode IDs.
    This is useful for collecting rollout data from vectorized environments
    where multiple episodes run simultaneously.
    
    Features:
        - Tracks episode IDs per environment to avoid mixing trajectories
        - Supports skipping already-done environments (for auto-reset handling)
        - Provides methods to retrieve complete episodes by ID
        - Stores raw (unnormalized) data for flexibility
    
    Example:
        >>> buffer = RolloutReplayBuffer(capacity=10000, num_envs=4)
        >>> # During rollout loop:
        >>> buffer.add_from_parallel_envs(
        ...     obs_batch=obs_dict,
        ...     action_batch=actions,
        ...     reward_batch=rewards,
        ...     next_obs_batch=next_obs_dict,
        ...     done_batch=dones,
        ...     skip_mask=env_done,  # Skip envs that already finished
        ... )
        >>> # Get complete episode:
        >>> episode = buffer.get_episode_by_id(0)
    """
    
    def __init__(
        self,
        capacity: int,
        num_envs: int = 1,
        chunk_size: int = 1,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        device: str = 'cuda:0',
        storage_device: str = 'cpu',
    ):
        """
        Initialize the rollout replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            num_envs: Number of parallel environments
            chunk_size: Action chunk size (default 1 for single-step actions)
            ctrl_space: Control space ('ee' or 'joint')
            ctrl_type: Control type ('delta' or 'abs')
            device: Device for sampling (e.g., 'cuda:0')
            storage_device: Device for storage (e.g., 'cpu')
        """
        super().__init__(
            capacity=capacity,
            chunk_size=chunk_size,
            action_normalizer=None,
            state_normalizer=None,
            transforms=None,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            device=device,
            storage_device=storage_device,
            store_raw=True,
        )
        
        self.num_envs = num_envs
        # Track episode IDs per environment
        # Use distinct episode ids for each env to avoid mixing trajectories
        self._env_episode_ids = list(range(num_envs))
        self._next_episode_id = num_envs  # Start from num_envs for next episodes
        
        # Episode tracking arrays (will be allocated on first add)
        self.episode_ids = None
        self.timestamps = None
        self._env_timesteps = [0] * num_envs  # Timestep counter per env
    
    def add_from_parallel_envs(
        self,
        obs_batch: Dict[str, np.ndarray],  # Batched observations {key: (num_envs, ...)}
        action_batch: np.ndarray,          # (num_envs, action_dim)
        reward_batch: np.ndarray,          # (num_envs,)
        next_obs_batch: Dict[str, np.ndarray],
        done_batch: np.ndarray,            # (num_envs,) boolean
        truncated_batch: Optional[np.ndarray] = None,
        info_batch: Optional[List[Dict]] = None,
        skip_mask: Optional[np.ndarray] = None,  # (num_envs,) boolean - True to skip
    ):
        """
        Add transitions from parallel environments.
        
        Separates batched data and assigns different episode IDs per environment.
        When an environment finishes (done=True), its next transitions will use
        a new episode ID.
        
        Args:
            obs_batch: Batched observations dict {key: (num_envs, ...)}
            action_batch: Batched actions (num_envs, action_dim)
            reward_batch: Rewards per env (num_envs,)
            next_obs_batch: Batched next observations dict
            done_batch: Done flags per env (num_envs,)
            truncated_batch: Truncated flags per env (optional)
            info_batch: Info dicts per env (optional)
            skip_mask: Boolean mask indicating which envs to skip (optional).
                       Use this to skip already-done environments when handling
                       auto-reset in vectorized environments.
        """
        num_envs = len(done_batch)
        if truncated_batch is None:
            truncated_batch = np.zeros(num_envs, dtype=bool)
        if skip_mask is None:
            skip_mask = np.zeros(num_envs, dtype=bool)
        
        # Pre-allocate episode tracking if not done
        if self.episode_ids is None:
            self.episode_ids = np.zeros(self.capacity, dtype=np.int64)
            self.timestamps = np.zeros(self.capacity, dtype=np.int64)
        
        # Add each environment's transition separately
        for env_idx in range(num_envs):
            # Skip already-done environments
            if skip_mask[env_idx]:
                continue
            
            # Extract single env data
            obs = self._extract_single_obs(obs_batch, env_idx)
            next_obs = self._extract_single_obs(next_obs_batch, env_idx)
            action = action_batch[env_idx]
            reward = float(reward_batch[env_idx])
            done = bool(done_batch[env_idx])
            truncated = bool(truncated_batch[env_idx])
            
            # Create MetaObs and MetaAction
            meta_obs = self._dict_to_metaobs(obs)
            meta_next_obs = self._dict_to_metaobs(next_obs)
            meta_action = MetaAction(
                action=action,
                ctrl_space=self.ctrl_space,
                ctrl_type=self.ctrl_type,
            )
            
            # Store episode_id and timestamp BEFORE adding
            current_ep_id = self._env_episode_ids[env_idx]
            current_ts = self._env_timesteps[env_idx]
            
            # Add to buffer (no normalization, raw data)
            self.add(
                obs=meta_obs,
                action=meta_action,
                reward=reward,
                next_obs=meta_next_obs,
                done=done,
                truncated=truncated,
                info=info_batch[env_idx] if info_batch is not None else None,
                already_normalized=True,  # Store raw data without normalizing
            )
            
            # Store episode metadata at the position we just wrote
            pos = (self.position - 1) % self.capacity
            self.episode_ids[pos] = current_ep_id
            self.timestamps[pos] = current_ts
            
            # Update episode_ends
            self.episode_ends[pos] = done or truncated
            
            # Update timestep counter
            self._env_timesteps[env_idx] += 1
            
            # If episode ended, assign new episode ID for this env
            if done or truncated:
                self._env_episode_ids[env_idx] = self._next_episode_id
                self._next_episode_id += 1
                self._env_timesteps[env_idx] = 0  # Reset timestep
    
    def _extract_single_obs(self, obs_batch: Dict[str, np.ndarray], env_idx: int) -> Dict[str, np.ndarray]:
        """Extract single environment's observation from batched observations."""
        single_obs = {}
        for key, value in obs_batch.items():
            if value is None:
                single_obs[key] = None
            elif isinstance(value, np.ndarray):
                single_obs[key] = value[env_idx]
            else:
                single_obs[key] = value
        return single_obs
    
    # get_episode_by_id and save_episode_as_video are inherited from ILReplayBuffer
    
    def get_all_episode_ids(self) -> List[int]:
        """Get list of all unique episode IDs in the buffer."""
        if self.episode_ids is None or self.size == 0:
            return []
        return list(np.unique(self.episode_ids[:self.size]))
    
    def get_episode_lengths(self) -> Dict[int, int]:
        """Get the length of each episode.
        
        Returns:
            Dict mapping episode_id to episode length
        """
        episode_ids = self.get_all_episode_ids()
        lengths = {}
        for ep_id in episode_ids:
            mask = self.episode_ids[:self.size] == ep_id
            lengths[ep_id] = int(np.sum(mask))
        return lengths
    
    def get_complete_episodes(self) -> List[int]:
        """Get IDs of episodes that are complete (have a done/truncated transition).
        
        Returns:
            List of episode IDs that have ended
        """
        if self.episode_ids is None or self.size == 0:
            return []
        
        complete_ids = []
        for ep_id in self.get_all_episode_ids():
            mask = self.episode_ids[:self.size] == ep_id
            indices = np.where(mask)[0]
            # Check if any transition in this episode has done=True
            if any(self.episode_ends[idx] for idx in indices):
                complete_ids.append(ep_id)
        
        return complete_ids
    
    def get_successful_episodes(self) -> List[int]:
        """Get IDs of episodes that ended with success (done=True, not truncated).
        
        Returns:
            List of successful episode IDs
        """
        if self.episode_ids is None or self.size == 0:
            return []
        
        successful_ids = []
        for ep_id in self.get_all_episode_ids():
            mask = self.episode_ids[:self.size] == ep_id
            indices = np.where(mask)[0]
            # Check if any transition has done=True and truncated=False
            for idx in indices:
                if self.dones[idx] and not self.truncateds[idx]:
                    successful_ids.append(ep_id)
                    break
        
        return successful_ids
    
    def reset_episode_tracking(self):
        """Reset episode tracking for new rollout collection.
        
        This resets the episode ID counters but does not clear the buffer.
        Useful when starting a new set of rollouts.
        """
        self._env_episode_ids = list(range(self.num_envs))
        self._next_episode_id = self.num_envs
        self._env_timesteps = [0] * self.num_envs

