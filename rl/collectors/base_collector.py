"""
Base Collector Class

This module defines the base class for all data collectors in the RL framework.

Design Philosophy:
- Responsibility separation: Separate data collection logic from trainer
  so that trainer can focus on training loop coordination
- Environment abstraction: Support single environment, parallel environments, multiple environment types
- Raw data storage: Only store raw environment rewards, no reward function computation,
  ensuring data integrity
- Statistics: Collect and return episode statistics
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction, MetaEnv


class BaseCollector(ABC):
    """
    Base class for data collectors.
    
    This class defines the interface for all data collectors in the RL framework.
    Collectors gather experience data from environments by interacting with them
    using the algorithm's policy.
    
    Attributes:
        meta_envs: Environment(s) to collect data from
        algorithm: Algorithm instance for action selection and transition recording
    
    Note: Collector only stores raw environment rewards, no reward function computation.
          Reward functions are applied in the trainer during training time.
    """
    
    def __init__(
        self,
        meta_envs: Union['MetaEnv', List['MetaEnv'], Callable, Dict[str, Any]],
        algorithm: 'BaseAlgorithm',
        **kwargs
    ):
        """
        Initialize the collector.
        
        Args:
            meta_envs: Supports multiple formats:
                      - MetaEnv instance: Single environment
                      - List[MetaEnv]: Environment list (same type environments)
                      - Callable: Environment factory function
                      - Dict[str, Any]: Multi-environment config dict (supports different env types)
            algorithm: BaseAlgorithm instance (required)
                      - Used for action selection and transition recording
            **kwargs: Collector-specific parameters
        
        Note: Collector only stores raw environment rewards, no reward function computation.
              Reward functions are applied in trainer during training time.
        """
        self.meta_envs = meta_envs
        self.algorithm = algorithm
        self._kwargs = kwargs
    
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
    
    def get_envs(self) -> Union['MetaEnv', List['MetaEnv'], Dict[str, Any]]:
        """Get the underlying environment(s)."""
        return self.meta_envs
    
    def get_algorithm(self) -> 'BaseAlgorithm':
        """Get the algorithm instance."""
        return self.algorithm
    
    def __repr__(self) -> str:
        env_info = type(self.meta_envs).__name__ if not isinstance(self.meta_envs, (list, dict)) else f"List[{len(self.meta_envs)}]" if isinstance(self.meta_envs, list) else f"Dict[{len(self.meta_envs)}]"
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
            self._max_steps = 100
        
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
    
    # Simple collector implementation for testing
    class SimpleCollector(BaseCollector):
        """Simple collector for testing."""
        
        def __init__(
            self,
            meta_envs: Union[MetaEnv, List[MetaEnv]],
            algorithm: BaseAlgorithm,
            **kwargs
        ):
            super().__init__(meta_envs, algorithm, **kwargs)
            
            # Initialize environments
            if isinstance(meta_envs, list):
                self.envs = meta_envs
            elif isinstance(meta_envs, MetaEnv):
                self.envs = [meta_envs]
            else:
                raise ValueError(f"Unsupported env type: {type(meta_envs)}")
            
            self._last_obs = None
            self._last_dones = None
            self._episode_rewards = []
            self._episode_lengths = []
            self._current_episode_reward = [0.0] * len(self.envs)
            self._current_episode_length = [0] * len(self.envs)
        
        def reset(self, **kwargs) -> None:
            """Reset all environments."""
            self._last_obs = []
            self._last_dones = []
            for env in self.envs:
                obs = env.reset()
                self._last_obs.append(obs)
                self._last_dones.append(False)
            self._current_episode_reward = [0.0] * len(self.envs)
            self._current_episode_length = [0] * len(self.envs)
        
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
            
            for step in range(n_steps):
                for i, (env, obs) in enumerate(zip(self.envs, self._last_obs)):
                    if self._last_dones[i]:
                        continue
                    
                    # Get action
                    with torch.no_grad():
                        action = self.algorithm.select_action(obs)
                    
                    # Environment interaction
                    new_obs, reward, done, info = env.step(action)
                    
                    # Record transition (only store raw reward)
                    transition_kwargs = {}
                    if env_type is not None:
                        transition_kwargs['env_type'] = env_type
                    
                    self.algorithm.record_transition(
                        state=obs,
                        action=action,
                        reward=reward,  # Store raw reward
                        next_state=new_obs,
                        done=done,
                        info=info,
                        **transition_kwargs
                    )
                    
                    # Update episode statistics
                    self._current_episode_reward[i] += reward
                    self._current_episode_length[i] += 1
                    
                    # If episode ended
                    if done:
                        stats['episode_rewards'].append(self._current_episode_reward[i])
                        stats['episode_lengths'].append(self._current_episode_length[i])
                        
                        # Reset environment and episode stats
                        new_obs = env.reset()
                        self._current_episode_reward[i] = 0.0
                        self._current_episode_length[i] = 0
                    
                    self._last_obs[i] = new_obs
                    self._last_dones[i] = done if not done else False  # Reset done flag after reset
                    stats['total_steps'] += 1
            
            return stats
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseCollector (SimpleCollector implementation)")
    print("=" * 60)
    
    # Create environment, policy, algorithm
    print("\n1. Creating components...")
    env = DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20)
    meta_policy = DummyMetaPolicy(action_dim=7)
    replay = SimpleReplay(capacity=1000)
    algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=replay)
    
    print(f"   Environment: {env}")
    print(f"   MetaPolicy: {meta_policy}")
    print(f"   Algorithm: {algorithm}")
    
    # Create collector
    print("\n2. Creating collector...")
    collector = SimpleCollector(
        meta_envs=env,
        algorithm=algorithm
    )
    print(f"   Collector: {collector}")
    
    # Test reset
    print("\n3. Testing reset...")
    collector.reset()
    print("   Reset successful")
    
    # Test collect
    print("\n4. Testing collect...")
    stats = collector.collect(n_steps=50)
    print(f"   Collected {stats['total_steps']} steps")
    print(f"   Episode rewards: {stats['episode_rewards']}")
    print(f"   Episode lengths: {stats['episode_lengths']}")
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Test with env_type
    print("\n5. Testing collect with env_type...")
    # Create algorithm with multi-replay
    multi_replay = {
        'indoor': SimpleReplay(capacity=1000),
        'outdoor': SimpleReplay(capacity=1000)
    }
    multi_algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=multi_replay)
    
    indoor_env = DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20)
    indoor_collector = SimpleCollector(
        meta_envs=indoor_env,
        algorithm=multi_algorithm
    )
    
    # Collect to indoor replay
    indoor_collector.reset()
    indoor_stats = indoor_collector.collect(n_steps=30, env_type='indoor')
    print(f"   Indoor collected {indoor_stats['total_steps']} steps")
    print(f"   Indoor replay size: {len(multi_algorithm.replay['indoor'])}")
    print(f"   Outdoor replay size: {len(multi_algorithm.replay['outdoor'])}")
    
    # Test with multiple environments
    print("\n6. Testing with multiple environments...")
    envs = [
        DummyMetaEnv(state_dim=10, action_dim=7, max_steps=15),
        DummyMetaEnv(state_dim=10, action_dim=7, max_steps=15),
    ]
    multi_env_collector = SimpleCollector(
        meta_envs=envs,
        algorithm=DummyAlgorithm(meta_policy=meta_policy, replay=SimpleReplay(capacity=1000))
    )
    multi_env_collector.reset()
    multi_env_stats = multi_env_collector.collect(n_steps=50)
    print(f"   Collected from {len(envs)} environments")
    print(f"   Total steps: {multi_env_stats['total_steps']}")
    print(f"   Episodes completed: {len(multi_env_stats['episode_rewards'])}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

