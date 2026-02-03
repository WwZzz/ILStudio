"""
Base Collector Class

This module defines the base class for all data collectors in the RL framework.

Design Philosophy:
- Responsibility separation: Separate data collection logic from trainer
  so that trainer can focus on training loop coordination
- Environment abstraction: Support vectorized environments (SequentialVectorEnv, SubprocVectorEnv, etc.)
- Raw data storage: Only store raw environment rewards, no reward function computation,
  ensuring data integrity
- Statistics: Collect and return episode statistics
- Exploration support: Support random exploration phase and noise-based exploration
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction

# Vector environment protocol
from rl.envs import VectorEnvProtocol, EnvsType

if TYPE_CHECKING:
    from utils.exploration import ExplorationScheduler


class BaseCollector(ABC):
    """
    Base class for data collectors.
    
    This class defines the interface for all data collectors in the RL framework.
    Collectors gather experience data from vectorized environments by interacting with them
    using the algorithm's policy.
    
    Attributes:
        envs: Vectorized environment(s) to collect data from
        algorithm: Algorithm instance for action selection and transition recording
        exploration: Optional exploration strategy for action exploration
    
    Note: Collector only stores raw environment rewards, no reward function computation.
          Reward functions are applied in the trainer during training time.
    """
    
    def __init__(
        self,
        envs: Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]],
        algorithm: 'BaseAlgorithm',
        exploration: Optional['ExplorationScheduler'] = None,
        **kwargs
    ):
        """
        Initialize the collector.
        
        Args:
            envs: Vectorized environment(s), supports:
                - VectorEnvProtocol: Single vectorized environment
                  (SequentialVectorEnv, SubprocVectorEnv, DummyVectorEnv, etc.)
                - Dict[str, VectorEnvProtocol]: Multi-environment dict for different env types
                  e.g., {'sim': sim_vec_env, 'real': real_vec_env}
            algorithm: BaseAlgorithm instance (required)
                      - Used for action selection and transition recording
            exploration: Optional ExplorationScheduler for action exploration
                        - If None, no exploration is applied (use policy actions directly)
                        - Supports initial random exploration + noise-based exploration
                        - Example:
                          exploration = ExplorationScheduler(
                              exploration_strategy=GaussianNoise(sigma=0.1),
                              random_steps=10000,
                              action_low=np.array([-1.0] * 7),
                              action_high=np.array([1.0] * 7),
                          )
            **kwargs: Collector-specific parameters
        
        Note: Collector only stores raw environment rewards, no reward function computation.
              Reward functions are applied in trainer during training time.
        """
        self.envs = envs
        self.algorithm = algorithm
        self.exploration = exploration
        self._kwargs = kwargs
        
        # Total steps counter for exploration scheduling
        self._total_steps = 0
        
        # Normalize environment storage: always use dict internally
        if isinstance(envs, dict):
            self._envs_dict: Dict[str, VectorEnvProtocol] = envs
            self._is_multi_env = True
        else:
            self._envs_dict = {'default': envs}
            self._is_multi_env = False
    
    @abstractmethod
    def collect(self, n_steps: int, env_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect n_steps of interaction data.
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional, environment type identifier (for multi-environment scenarios)
                     - If provided, will be passed to record_transition with env_type
                     - Used to support a single algorithm storing data from multiple different environments
        
        Returns:
            Dictionary containing statistics, such as:
            - episode_rewards: List of episode rewards
            - episode_lengths: List of episode lengths
            - total_steps: Total number of steps collected
            - env_type_stats: Statistics grouped by environment type (if multi-environment is supported)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, **kwargs) -> None:
        """
        Reset collector state (e.g., reset environments).
        
        Args:
            **kwargs: Reset parameters
        """
        raise NotImplementedError
    
    def get_env(self, env_type: Optional[str] = None) -> VectorEnvProtocol:
        """
        Get the vectorized environment by type.
        
        Args:
            env_type: Environment type identifier. If None, returns 'default' env
                     or the first env if 'default' doesn't exist.
        
        Returns:
            The vectorized environment
        
        Raises:
            KeyError: If specified env_type is not found
        """
        if env_type is not None:
            return self._envs_dict[env_type]
        
        # Return 'default' if exists, otherwise return first env
        if 'default' in self._envs_dict:
            return self._envs_dict['default']
        return list(self._envs_dict.values())[0]
    
    def get_total_env_num(self) -> int:
        """
        Get total number of parallel environments across all types.
        
        Returns:
            Total count of parallel environments
        """
        return sum(len(env) for env in self._envs_dict.values())
    
    def get_env_types(self) -> List[str]:
        """
        Get all environment type identifiers.
        
        Returns:
            List of environment type strings
        """
        return list(self._envs_dict.keys())
    
    def get_envs(self) -> Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]]:
        """Get the underlying environment(s)."""
        return self.envs
    
    # ==================== Exploration Methods ====================
    
    def set_exploration(self, exploration: Optional['ExplorationScheduler']) -> None:
        """
        Set or update the exploration strategy.
        
        Args:
            exploration: ExplorationScheduler instance or None to disable exploration
        """
        self.exploration = exploration
    
    def apply_exploration(
        self, 
        action: Any, 
        obs: Any = None,
        **kwargs
    ) -> Any:
        """
        Apply exploration to action(s).
        
        Supports:
        - MetaAction: Single action or batched (action.action shape: (action_dim,) or (n_envs, action_dim))
        - List[MetaAction]: List of individual actions
        - np.ndarray: Raw action array
        - dict with 'action' key
        
        Args:
            action: Action(s) from policy
            obs: Optional observation (for uncertainty-based exploration)
            **kwargs: Additional arguments for exploration strategy
        
        Returns:
            Explored action(s) (same type/structure as input)
        """
        if self.exploration is None:
            return action
        
        # Case 1: List of MetaAction objects
        if isinstance(action, list) and len(action) > 0 and hasattr(action[0], 'action'):
            # Stack actions, apply exploration, then update each
            action_arrays = [a.action for a in action]
            stacked = np.stack(action_arrays)  # (n_envs, action_dim)
            explored = self.exploration(
                stacked,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            # Update each MetaAction
            for i, a in enumerate(action):
                a.action = explored[i]
            return action
        
        # Case 2: Single MetaAction (possibly with batched action field)
        elif hasattr(action, 'action') and action.action is not None:
            action_array = action.action
            explored_array = self.exploration(
                action_array, 
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            action.action = explored_array
            return action
        
        # Case 3: Dict with 'action' key
        elif isinstance(action, dict) and 'action' in action:
            action_array = action['action']
            explored_array = self.exploration(
                action_array,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
            action['action'] = explored_array
            return action
        
        # Case 4: Raw numpy array
        elif isinstance(action, np.ndarray):
            return self.exploration(
                action,
                step=self._total_steps,
                obs=obs,
                **kwargs
            )
        
        # Case 5: Unknown type, return as-is
        else:
            return action
    
    def reset_exploration(self, env_idx: Optional[int] = None) -> None:
        """
        Reset exploration state (e.g., for OU noise).
        
        Args:
            env_idx: Optional specific environment index to reset
        """
        if self.exploration is not None:
            self.exploration.reset(env_idx)
    
    @property
    def total_steps(self) -> int:
        """Get total steps collected so far."""
        return self._total_steps
    
    @property
    def is_exploring_randomly(self) -> bool:
        """Check if currently in random exploration phase."""
        if self.exploration is None:
            return False
        return self.exploration.is_random_phase
    
    def _unpack_actions(self, actions: Any, n_envs: int) -> List[Any]:
        """
        Unpack batched actions into a list for vec_env.step.
        
        Handles:
        - MetaAction with batched action field: (n_envs, action_dim) -> List[MetaAction]
        - List[MetaAction]: Return as-is
        - np.ndarray: (n_envs, action_dim) -> List[np.ndarray]
        
        Args:
            actions: Batched actions from policy
            n_envs: Number of environments
        
        Returns:
            List of individual actions, one per environment
        """
        # Case 1: Already a list
        if isinstance(actions, list):
            return actions
        
        # Case 2: MetaAction with batched action field
        if hasattr(actions, 'action') and actions.action is not None:
            action_array = actions.action
            if isinstance(action_array, np.ndarray) and action_array.ndim >= 2:
                # Batched: (n_envs, action_dim) or (n_envs, chunk_size, action_dim)
                unpacked = []
                for i in range(min(n_envs, len(action_array))):
                    # Create new MetaAction for each env
                    single_action = MetaAction(
                        action=action_array[i],
                        ctrl_space=getattr(actions, 'ctrl_space', 'ee'),
                        ctrl_type=getattr(actions, 'ctrl_type', 'delta'),
                        gripper_continuous=getattr(actions, 'gripper_continuous', False),
                    )
                    unpacked.append(single_action)
                return unpacked
            else:
                # Single action, replicate for all envs
                return [actions] * n_envs
        
        # Case 3: np.ndarray (batched)
        if isinstance(actions, np.ndarray):
            if actions.ndim >= 2:
                return [actions[i] for i in range(min(n_envs, len(actions)))]
            else:
                return [actions] * n_envs
        
        # Case 4: Unknown, replicate for all envs
        return [actions] * n_envs
    
    def get_algorithm(self) -> 'BaseAlgorithm':
        """Get the algorithm instance."""
        return self.algorithm
    
    @property
    def env_num(self) -> int:
        """
        Number of environments in the default (or first) environment.
        
        Returns:
            Number of parallel environments
        """
        if 'default' in self._envs_dict:
            return len(self._envs_dict['default'])
        return len(list(self._envs_dict.values())[0])
    
    def __repr__(self) -> str:
        env_info = f"{self.get_total_env_num()} envs" if self._is_multi_env else f"{self.env_num} envs"
        return f"{self.__class__.__name__}(envs={env_info}, algorithm={self.algorithm.__class__.__name__})"


class DummyCollector(BaseCollector):
    """
    Simple collector implementation for testing and basic use cases.
    
    This collector:
    - Works with vectorized environments (SequentialVectorEnv, SubprocVectorEnv, etc.)
    - Handles batched observations and actions
    - Creates RLTransition objects for storage in replay buffer
    - Tracks episode statistics
    """
    
    def __init__(self, envs, algorithm, **kwargs):
        super().__init__(envs, algorithm, **kwargs)
        self.vec_env = self.get_env()
        self._last_obs = None
    
    def reset(self, **kwargs):
        """Reset the collector and environments."""
        self.vec_env = self.get_env()
        self._last_obs = self.vec_env.reset()
    
    def collect(self, n_steps, env_type=None):
        """
        Collect n_steps of interaction data.
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional environment type identifier
        
        Returns:
            Dictionary with statistics:
            - episode_rewards: List of episode rewards
            - episode_lengths: List of episode lengths
            - total_steps: Total number of steps collected
            - env_type: Environment type identifier
        """
        if self._last_obs is None:
            self.reset()
        
        from benchmark.base import dict2meta, MetaObs, MetaAction
        from benchmark.utils import organize_obs
        from rl.buffer.transition import RLTransition
        
        stats = {'episode_rewards': [], 'episode_lengths': [], 'total_steps': 0, 'env_type': env_type}
        episode_rewards = np.zeros(len(self.vec_env))
        episode_lengths = np.zeros(len(self.vec_env), dtype=int)
        
        for step in range(n_steps):
            # Organize observations into batched MetaObs
            batched_obs = organize_obs(self._last_obs) if not isinstance(self._last_obs, MetaObs) else self._last_obs
            
            # Get batched actions for all environments at once
            # actions is expected to be an object array of dicts (from MetaPolicy)
            actions = self.algorithm.select_action(batched_obs)
            
            # Apply exploration if configured
            if self.exploration is not None:
                if self.is_exploring_randomly:
                    stats['random_steps'] = stats.get('random_steps', 0) + len(self.vec_env)
                actions = self.apply_exploration(actions, obs=batched_obs)
            
            # Step all environments (expects array of dicts, one per env)
            new_obs, rewards, dones, infos = self.vec_env.step(actions)
            
            # Organize next observations into batched MetaObs
            batched_next_obs = organize_obs(new_obs) if not isinstance(new_obs, MetaObs) else new_obs
            
            # Reconstruct MetaAction for storage (since actions is now an object array)
            # We need to extract the raw action arrays from the dicts
            if isinstance(actions, np.ndarray) and actions.dtype == object:
                # Extract 'action' field from each dict
                raw_actions = np.stack([a['action'] for a in actions])
                # Create MetaAction
                stored_actions = MetaAction(action=raw_actions)
            elif isinstance(actions, MetaAction):
                stored_actions = actions
            else:
                stored_actions = dict2meta(actions, mtype='act')
            
            # Create single RLTransition for all environments
            truncated = np.array([infos[i].get('TimeLimit.truncated', False) for i in range(len(infos))]) if infos else np.zeros_like(dones)
            
            transition = RLTransition(
                obs=batched_obs,
                action=stored_actions,
                next_obs=batched_next_obs,
                reward=rewards,
                done=dones,
                truncated=truncated,
                info=infos if infos else None,
            )
            
            kwargs_trans = {'env_type': env_type} if env_type else {}
            self.algorithm.record_transition(transition, **kwargs_trans)
            
            # Update episode statistics
            episode_rewards += rewards
            episode_lengths += 1
            stats['total_steps'] += len(self.vec_env)
            self._total_steps += len(self.vec_env)
            
            # Handle done environments
            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                stats['episode_rewards'].extend(episode_rewards[done_indices].tolist())
                stats['episode_lengths'].extend(episode_lengths[done_indices].tolist())
                episode_rewards[done_indices] = 0.0
                episode_lengths[done_indices] = 0
                # Reset done environments
                for idx in done_indices:
                    new_obs[idx] = self.vec_env.reset(id=idx)
            
            # Reorganize observations for next iteration
            self._last_obs = organize_obs(new_obs) if not isinstance(new_obs, MetaObs) else new_obs
        
        return stats

