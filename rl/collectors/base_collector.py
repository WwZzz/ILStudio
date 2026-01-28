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
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction

# Vector environment protocol
from rl.envs import VectorEnvProtocol, EnvsType


class BaseCollector(ABC):
    """
    Base class for data collectors.
    
    This class defines the interface for all data collectors in the RL framework.
    Collectors gather experience data from vectorized environments by interacting with them
    using the algorithm's policy.
    
    Attributes:
        envs: Vectorized environment(s) to collect data from
        algorithm: Algorithm instance for action selection and transition recording
    
    Note: Collector only stores raw environment rewards, no reward function computation.
          Reward functions are applied in the trainer during training time.
    """
    
    def __init__(
        self,
        envs: Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]],
        algorithm: 'BaseAlgorithm',
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
            **kwargs: Collector-specific parameters
        
        Note: Collector only stores raw environment rewards, no reward function computation.
              Reward functions are applied in trainer during training time.
        """
        self.envs = envs
        self.algorithm = algorithm
        self._kwargs = kwargs
        
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


if __name__ == '__main__':
    """
    Test code for BaseCollector class.
    
    Since BaseCollector is abstract, we create a simple concrete implementation for testing.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    import torch
    from benchmark.base import MetaObs, MetaAction, MetaEnv, MetaPolicy
    from rl.buffer.base_replay import BaseReplay
    from rl.base import BaseAlgorithm
    from dataclasses import asdict
    
    # Simple replay buffer for testing
    class SimpleReplay(BaseReplay):
        def __init__(self, capacity=1000, device='cpu', **kwargs):
            super().__init__(capacity=capacity, device=device, **kwargs)
            self._storage = []
        
        def add(self, transition):
            if self._size < self.capacity:
                self._storage.append(transition)
                self._size += 1
            else:
                self._storage[self._position] = transition
            self._position = (self._position + 1) % self.capacity
        
        def sample(self, batch_size):
            if self._size == 0:
                return {}
            indices = np.random.randint(0, self._size, size=min(batch_size, self._size))
            return {
                'states': [self._storage[i]['state'] for i in indices],
                'actions': [self._storage[i]['action'] for i in indices],
                'rewards': np.array([self._storage[i]['reward'] for i in indices]),
            }
        
        def clear(self):
            self._storage = []
            self._size = 0
            self._position = 0
        
        def save(self, path, **kwargs):
            pass
        
        def load(self, path, **kwargs):
            pass
    
    # Simple dummy environment for testing
    class DummyEnv:
        def __init__(self, state_dim=10, action_dim=7):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self._step_count = 0
            self._max_steps = 5
        
        def reset(self):
            self._step_count = 0
            return {'state': np.random.randn(self.state_dim).astype(np.float32)}
        
        def step(self, action):
            self._step_count += 1
            obs = {'state': np.random.randn(self.state_dim).astype(np.float32)}
            reward = np.random.randn()
            done = self._step_count >= self._max_steps
            info = {'step': self._step_count}
            if done:
                info['episode'] = {'r': np.random.randn() * 10, 'l': self._step_count}
            return obs, reward, done, info
        
        def close(self):
            pass
    
    # Simple MetaEnv wrapper for testing
    class DummyMetaEnv(MetaEnv):
        def __init__(self, state_dim=10, action_dim=7, max_steps=100):
            self.env = DummyEnv(state_dim=state_dim, action_dim=action_dim)
            self.env._max_steps = max_steps
            self.prev_obs = None
        
        def obs2meta(self, raw_obs):
            return MetaObs(
                state=raw_obs['state'],
                raw_lang="test instruction"
            )
        
        def meta2act(self, action, *args):
            if hasattr(action, 'action'):
                return action.action
            return action
        
        def reset(self):
            init_obs = self.env.reset()
            self.prev_obs = self.obs2meta(init_obs)
            return self.prev_obs
        
        def step(self, action):
            act = self.meta2act(action)
            obs, reward, done, info = self.env.step(act)
            self.prev_obs = self.obs2meta(obs)
            return self.prev_obs, reward, done, info
    
    # Simple policy for testing
    class DummyPolicy:
        def __init__(self, action_dim=7):
            self.action_dim = action_dim
        
        def select_action(self, obs):
            return MetaAction(
                action=np.random.randn(self.action_dim).astype(np.float32),
                ctrl_space='ee',
                ctrl_type='delta'
            )
        
        def train(self):
            pass
        
        def eval(self):
            pass
    
    # Simple MetaPolicy wrapper for testing
    class DummyMetaPolicy(MetaPolicy):
        def __init__(self, action_dim=7):
            self.policy = DummyPolicy(action_dim=action_dim)
            self.chunk_size = 1
            self.ctrl_space = 'ee'
            self.ctrl_type = 'delta'
            self.action_queue = []
            self.action_normalizer = None
            self.state_normalizer = None
        
        def select_action(self, mobs, t=0, **kwargs):
            return self.policy.select_action(mobs)
    
    # Simple algorithm for testing
    class DummyAlgorithm(BaseAlgorithm):
        def __init__(self, meta_policy, replay=None, **kwargs):
            super().__init__(meta_policy=meta_policy, replay=replay, **kwargs)
            self._timestep = 0
        
        def update(self, batch=None, **kwargs):
            return {'loss': 0.0}
        
        def select_action(self, obs, **kwargs):
            return self.meta_policy.select_action(obs, t=self._timestep)
    
    # Simple VectorEnv for testing (similar to SequentialVectorEnv)
    class DummyVectorEnv:
        """Simple sequential vector environment for testing."""
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]
            self.env_num = len(self.envs)
        
        def reset(self, id=None):
            if id is None:
                obs_list = [env.reset() for env in self.envs]
                return np.array(obs_list, dtype=object)
            else:
                if np.isscalar(id):
                    return self.envs[id].reset()
                else:
                    obs_list = [self.envs[i].reset() for i in id]
                    return np.array(obs_list, dtype=object)
        
        def step(self, action, id=None):
            if id is None:
                results = [env.step(act) for env, act in zip(self.envs, action)]
                obs = np.array([r[0] for r in results], dtype=object)
                rew = np.array([r[1] for r in results])
                done = np.array([r[2] for r in results])
                info = [r[3] for r in results]
                return obs, rew, done, info
            else:
                if np.isscalar(id):
                    return self.envs[id].step(action)
                else:
                    results = [self.envs[i].step(act) for i, act in zip(id, action)]
                    obs = np.array([r[0] for r in results], dtype=object)
                    rew = np.array([r[1] for r in results])
                    done = np.array([r[2] for r in results])
                    info = [r[3] for r in results]
                    return obs, rew, done, info
        
        def close(self):
            for env in self.envs:
                if hasattr(env, 'close'):
                    env.close()
        
        def __len__(self):
            return self.env_num
    
    # Simple collector implementation for testing
    class SimpleCollector(BaseCollector):
        """Simple collector for testing with vectorized environments."""
        
        def __init__(
            self,
            envs: Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]],
            algorithm: BaseAlgorithm,
            **kwargs
        ):
            super().__init__(envs, algorithm, **kwargs)
            
            # Get default environment
            self.vec_env = self.get_env()
            self._last_obs = None
            self._last_dones = None
            self._current_episode_reward = None
            self._current_episode_length = None
        
        def reset(self, **kwargs) -> None:
            """Reset all environments."""
            self._last_obs = self.vec_env.reset()
            self._last_dones = np.zeros(len(self.vec_env), dtype=bool)
            self._current_episode_reward = np.zeros(len(self.vec_env), dtype=np.float32)
            self._current_episode_length = np.zeros(len(self.vec_env), dtype=int)
        
        def collect(self, n_steps: int, env_type: Optional[str] = None) -> Dict[str, Any]:
            """
            Collect n_steps of interaction data.
            
            Args:
                n_steps: Number of steps to collect
                env_type: Optional environment type identifier
            
            Returns:
                Statistics dictionary
            """
            if self._last_obs is None:
                self.reset()
            
            stats = {
                'episode_rewards': [],
                'episode_lengths': [],
                'total_steps': 0
            }
            
            steps_executed = 0
            
            while steps_executed < n_steps:
                # Get actions for all environments
                actions = []
                for obs in self._last_obs:
                    with torch.no_grad():
                        action = self.algorithm.select_action(obs)
                    actions.append(action)
                
                # Step all environments
                new_obs, rewards, dones, infos = self.vec_env.step(actions)
                
                # Record transitions and update stats
                for i in range(len(self.vec_env)):
                    # Record transition (only store raw reward)
                    transition_kwargs = {}
                    if env_type is not None:
                        transition_kwargs['env_type'] = env_type
                    
                    self.algorithm.record_transition(
                        state=self._last_obs[i],
                        action=actions[i],
                        reward=rewards[i],  # Store raw reward
                        next_state=new_obs[i],
                        done=dones[i],
                        info=infos[i],
                        **transition_kwargs
                    )
                    
                    # Update episode statistics
                    self._current_episode_reward[i] += rewards[i]
                    self._current_episode_length[i] += 1
                    stats['total_steps'] += 1
                    steps_executed += 1
                
                # Reset done environments
                done_indices = np.where(dones)[0]
                if len(done_indices) > 0:
                    for idx in done_indices:
                        stats['episode_rewards'].append(self._current_episode_reward[idx])
                        stats['episode_lengths'].append(self._current_episode_length[idx])
                        self._current_episode_reward[idx] = 0.0
                        self._current_episode_length[idx] = 0
                    
                    # Reset done environments
                    reset_obs = self.vec_env.reset(id=done_indices.tolist())
                    if len(done_indices) == 1:
                        new_obs[done_indices[0]] = reset_obs
                    else:
                        for idx, reset_idx in enumerate(done_indices):
                            new_obs[reset_idx] = reset_obs[idx]
                    
                    # Mark reset environments as not done
                    dones[done_indices] = False
                
                self._last_obs = new_obs
                self._last_dones = dones
            
            return stats
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseCollector with Vectorized Environments")
    print("=" * 60)
    
    # Create environment, policy, algorithm
    print("\n1. Creating components...")
    # Create vectorized environment
    # Use max_steps=5 to match DummyEnv's _max_steps=5 (user changed it)
    env_fn = lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=5)
    env_fns = [env_fn for _ in range(4)]
    vec_env = DummyVectorEnv(env_fns)
    
    meta_policy = DummyMetaPolicy(action_dim=7)
    replay = SimpleReplay(capacity=1000)
    algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=replay)
    
    print(f"   Vectorized Environment: {len(vec_env)} parallel envs")
    print(f"   MetaPolicy: {meta_policy}")
    print(f"   Algorithm: {algorithm}")
    
    # Create collector
    print("\n2. Creating collector...")
    collector = SimpleCollector(
        envs=vec_env,
        algorithm=algorithm
    )
    print(f"   Collector: {collector}")
    print(f"   env_num: {collector.env_num}")
    print(f"   total_env_num: {collector.get_total_env_num()}")
    
    # Test reset
    print("\n3. Testing reset...")
    collector.reset()
    print("   Reset successful")
    
    # Test collect
    print("\n4. Testing collect...")
    stats = collector.collect(n_steps=50)
    print(f"   Collected {stats['total_steps']} steps")
    print(f"   Episodes completed: {len(stats['episode_rewards'])}")
    print(f"   Episode rewards: {stats['episode_rewards'][:5]}...")  # Show first 5
    print(f"   Episode lengths: {stats['episode_lengths'][:5]}...")  # Show first 5
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Test with env_type
    print("\n5. Testing collect with env_type...")
    # Create algorithm with multi-replay
    multi_replay = {
        'indoor': SimpleReplay(capacity=1000),
        'outdoor': SimpleReplay(capacity=1000)
    }
    multi_algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=multi_replay)
    
    indoor_env_fn = lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20)
    indoor_vec_env = DummyVectorEnv([indoor_env_fn for _ in range(2)])
    indoor_collector = SimpleCollector(
        envs=indoor_vec_env,
        algorithm=multi_algorithm
    )
    
    # Collect to indoor replay
    indoor_collector.reset()
    indoor_stats = indoor_collector.collect(n_steps=30, env_type='indoor')
    print(f"   Indoor collected {indoor_stats['total_steps']} steps")
    print(f"   Indoor replay size: {len(multi_algorithm.replay['indoor'])}")
    print(f"   Outdoor replay size: {len(multi_algorithm.replay['outdoor'])}")
    
    # Test with multiple environment types (dict)
    print("\n6. Testing with multiple environment types (dict)...")
    sim_env_fn = lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=10)
    real_env_fn = lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=10)
    
    multi_envs = {
        'sim': DummyVectorEnv([sim_env_fn for _ in range(4)]),
        'real': DummyVectorEnv([real_env_fn for _ in range(2)]),
    }
    
    multi_env_collector = SimpleCollector(
        envs=multi_envs,
        algorithm=DummyAlgorithm(meta_policy=meta_policy, replay=SimpleReplay(capacity=1000))
    )
    print(f"   Multi-env collector: {multi_env_collector}")
    print(f"   env_types: {multi_env_collector.get_env_types()}")
    print(f"   total_env_num: {multi_env_collector.get_total_env_num()}")
    print(f"   sim env_num: {len(multi_env_collector.get_env('sim'))}")
    print(f"   real env_num: {len(multi_env_collector.get_env('real'))}")
    
    multi_env_collector.reset()
    multi_env_stats = multi_env_collector.collect(n_steps=1000)
    print(f"   Collected {multi_env_stats['total_steps']} steps")
    print(f"   Episodes completed: {len(multi_env_stats['episode_rewards'])}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

