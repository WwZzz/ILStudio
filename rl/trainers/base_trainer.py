"""
Base Trainer Class

This module defines the base class for all RL trainers in the framework.

Design Philosophy:
- Coordinate environment, policy, and algorithm for executing training loop
- Support single algorithm and multiple algorithms training scenarios
- Support custom reward functions (applied during training, not data collection)
- Support evaluation during training
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List, Callable
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction, MetaEnv


class BaseTrainer(ABC):
    """
    Base class for RL trainers.
    
    This class defines the interface for all trainers in the RL framework.
    Trainers coordinate the training loop by managing:
    - Environments (meta_envs)
    - Algorithms (with their policies and replay buffers)
    - Collectors (for data collection)
    - Reward functions (applied during training)
    
    Attributes:
        meta_envs: Environment(s) for training
        algorithm: Algorithm(s) for training
        collector: Data collector(s)
        reward_fn: Optional custom reward function
    """
    
    def __init__(
        self,
        meta_envs: Union[MetaEnv, List[MetaEnv], Callable, Dict[str, Any]],
        algorithm: Union['BaseAlgorithm', List['BaseAlgorithm']],
        collector: Optional[Union['BaseCollector', List['BaseCollector']]] = None,
        reward_fn: Optional[Union['BaseReward', Callable]] = None,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            meta_envs: Supports multiple formats:
                      - MetaEnv instance: Single environment
                      - List[MetaEnv]: Environment list (same type environments)
                      - Callable: Environment factory function
                      - Dict[str, Any]: Multi-environment config dict (different env types)
            algorithm: BaseAlgorithm instance or list of BaseAlgorithm (required)
                      - Single algorithm: Single agent training
                      - Algorithm list: Multiple algorithms training independently in same env
                        (each algorithm has its own replay buffer)
            collector: Optional data collector (if None, trainer creates default collector)
                      - Single algorithm: Single collector
                      - Multiple algorithms: Can be collector list, each for one algorithm
                      - If None, trainer creates default collector for each algorithm
            reward_fn: Optional reward function (if None, use raw environment reward)
                      - Can be BaseReward instance or Callable function
                      - Applied during training for algorithm updates
                      - Note: Replay buffer stores raw rewards, reward function only applied during training
            **kwargs: Trainer-specific parameters
        """
        self.meta_envs = meta_envs
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self._kwargs = kwargs
        
        # Handle collector initialization
        self.collector = collector
        # Note: Actual collector creation should be done in subclasses
        # since it may require specific collector types
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        """
        Execute training loop.
        
        Args:
            **kwargs: Training parameters, can include:
                - total_steps: Total training steps (optional)
                - total_episodes: Total episodes (optional)
                - max_time: Maximum training time (optional)
                - log_interval: Logging interval (optional)
                - save_interval: Model save interval (optional)
                - eval_interval: Evaluation interval (optional)
        """
        raise NotImplementedError
    
    def compute_reward(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward (support custom reward function).
        
        Used during training for algorithm update reward computation.
        Replay buffer stores raw rewards, reward function only applied during training.
        
        Args:
            state: Current state
            action: Action
            next_state: Next state
            env_reward: Environment raw reward
            info: Additional information dictionary
        
        Returns:
            Computed reward value
        """
        if self.reward_fn is not None:
            # Import here to avoid circular imports
            from rl.rewards.base_reward import BaseReward
            
            if isinstance(self.reward_fn, BaseReward):
                return self.reward_fn.compute(state, action, next_state, env_reward, info)
            else:
                # Assume it's a callable
                return self.reward_fn(state, action, next_state, env_reward, info)
        return env_reward
    
    def collect_rollout(
        self, 
        n_steps: int, 
        env_type: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Collect rollout data (using collector).
        
        Args:
            n_steps: Number of steps to collect
            env_type: Optional environment type identifier (for multi-environment scenarios)
                     - Used to support a single algorithm storing data from multiple different environments
        
        Returns:
            - Single algorithm: Rollout statistics dictionary
            - Multiple algorithms: List of rollout statistics dictionaries
        """
        if self.collector is None:
            raise ValueError("Collector is not initialized. Please provide or create a collector.")
        
        if isinstance(self.collector, list):
            # Multiple algorithms: Each algorithm collects independently
            return [col.collect(n_steps, env_type=env_type) for col in self.collector]
        else:
            # Single algorithm
            return self.collector.collect(n_steps, env_type=env_type)
    
    @abstractmethod
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        env_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate policy performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment (optional)
            env_type: Optional, specify evaluation environment type
            **kwargs: Other evaluation parameters
        
        Returns:
            Dictionary containing evaluation metrics
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model and training state."""
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model and training state."""
        raise NotImplementedError
    
    def get_algorithm(self) -> Union['BaseAlgorithm', List['BaseAlgorithm']]:
        """Get the algorithm(s)."""
        return self.algorithm
    
    def get_collector(self) -> Optional[Union['BaseCollector', List['BaseCollector']]]:
        """Get the collector(s)."""
        return self.collector
    
    def set_reward_fn(self, reward_fn: Union['BaseReward', Callable]) -> None:
        """
        Set or update the reward function.
        
        Args:
            reward_fn: New reward function to use
        """
        self.reward_fn = reward_fn
    
    def __repr__(self) -> str:
        algo_info = self.algorithm.__class__.__name__ if not isinstance(self.algorithm, list) else f"List[{len(self.algorithm)}]"
        collector_info = "None" if self.collector is None else (
            self.collector.__class__.__name__ if not isinstance(self.collector, list) else f"List[{len(self.collector)}]"
        )
        reward_info = "None" if self.reward_fn is None else self.reward_fn.__class__.__name__
        return f"{self.__class__.__name__}(algorithm={algo_info}, collector={collector_info}, reward_fn={reward_info})"


if __name__ == '__main__':
    """
    Test code for BaseTrainer class.
    
    Since BaseTrainer is abstract, we create a simple concrete implementation for testing.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction, MetaEnv, MetaPolicy
    from rl.buffer.base_replay import BaseReplay
    from rl.base import BaseAlgorithm
    from rl.rewards.base_reward import BaseReward, IdentityReward, ScaledReward
    from rl.collectors.base_collector import BaseCollector
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
        def __init__(self, state_dim=10, action_dim=7, max_steps=100):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self._step_count = 0
            self._max_steps = max_steps
        
        def reset(self):
            self._step_count = 0
            return {'state': np.random.randn(self.state_dim).astype(np.float32)}
        
        def step(self, action):
            self._step_count += 1
            obs = {'state': np.random.randn(self.state_dim).astype(np.float32)}
            reward = np.random.randn()
            done = self._step_count >= self._max_steps
            info = {'step': self._step_count}
            return obs, reward, done, info
        
        def close(self):
            pass
    
    # Simple MetaEnv wrapper for testing
    class DummyMetaEnv(MetaEnv):
        def __init__(self, state_dim=10, action_dim=7, max_steps=100):
            self.env = DummyEnv(state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
            self.prev_obs = None
        
        def obs2meta(self, raw_obs):
            return MetaObs(state=raw_obs['state'], raw_lang="test")
        
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
            return MetaAction(action=np.random.randn(self.action_dim).astype(np.float32))
        
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
            self._update_count = 0
        
        def update(self, batch=None, **kwargs):
            self._update_count += 1
            batch_size = kwargs.get('batch_size', 32)
            if batch is None and self.replay is not None:
                batch = self.replay.sample(batch_size)
            return {'loss': np.random.randn(), 'update_count': self._update_count}
        
        def select_action(self, obs, **kwargs):
            return self.meta_policy.select_action(obs, t=self._timestep)
    
    # Simple collector for testing
    class DummyCollector(BaseCollector):
        def __init__(self, meta_envs, algorithm, **kwargs):
            super().__init__(meta_envs, algorithm, **kwargs)
            if isinstance(meta_envs, list):
                self.envs = meta_envs
            else:
                self.envs = [meta_envs]
            self._last_obs = None
        
        def reset(self, **kwargs):
            self._last_obs = [env.reset() for env in self.envs]
        
        def collect(self, n_steps, env_type=None):
            if self._last_obs is None:
                self.reset()
            
            stats = {'episode_rewards': [], 'episode_lengths': [], 'total_steps': 0}
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(n_steps):
                for i, (env, obs) in enumerate(zip(self.envs, self._last_obs)):
                    action = self.algorithm.select_action(obs)
                    new_obs, reward, done, info = env.step(action)
                    
                    # Record transition
                    kwargs_trans = {'env_type': env_type} if env_type else {}
                    self.algorithm.record_transition(
                        state=obs, action=action, reward=reward,
                        next_state=new_obs, done=done, info=info, **kwargs_trans
                    )
                    
                    episode_reward += reward
                    episode_length += 1
                    stats['total_steps'] += 1
                    
                    if done:
                        stats['episode_rewards'].append(episode_reward)
                        stats['episode_lengths'].append(episode_length)
                        episode_reward = 0.0
                        episode_length = 0
                        new_obs = env.reset()
                    
                    self._last_obs[i] = new_obs
            
            return stats
    
    # Simple trainer implementation for testing
    class SimpleTrainer(BaseTrainer):
        """Simple trainer for testing."""
        
        def __init__(
            self,
            meta_envs,
            algorithm,
            collector=None,
            reward_fn=None,
            **kwargs
        ):
            super().__init__(meta_envs, algorithm, collector, reward_fn, **kwargs)
            
            # Create default collector if not provided
            if self.collector is None:
                if isinstance(algorithm, list):
                    self.collector = [
                        DummyCollector(meta_envs=meta_envs, algorithm=alg)
                        for alg in algorithm
                    ]
                else:
                    self.collector = DummyCollector(
                        meta_envs=meta_envs,
                        algorithm=algorithm
                    )
            
            self._total_steps = 0
            self._training_logs = []
        
        def train(self, **kwargs):
            total_steps = kwargs.get('total_steps', 1000)
            log_interval = kwargs.get('log_interval', 100)
            update_interval = kwargs.get('update_interval', 50)
            batch_size = kwargs.get('batch_size', 32)
            
            print(f"Starting training for {total_steps} steps...")
            
            while self._total_steps < total_steps:
                # Collect data
                stats = self.collect_rollout(n_steps=update_interval)
                self._total_steps += stats['total_steps'] if isinstance(stats, dict) else sum(s['total_steps'] for s in stats)
                
                # Update algorithm
                if isinstance(self.algorithm, list):
                    for alg in self.algorithm:
                        result = alg.update(batch_size=batch_size)
                else:
                    result = self.algorithm.update(batch_size=batch_size)
                
                # Log
                if self._total_steps % log_interval == 0:
                    print(f"  Step {self._total_steps}: loss={result.get('loss', 0.0):.4f}")
                    self._training_logs.append({
                        'step': self._total_steps,
                        'loss': result.get('loss', 0.0)
                    })
            
            print(f"Training completed. Total steps: {self._total_steps}")
        
        def evaluate(self, n_episodes=10, render=False, env_type=None, **kwargs):
            print(f"Evaluating for {n_episodes} episodes...")
            
            # Create eval env
            if isinstance(self.meta_envs, dict):
                env = list(self.meta_envs.values())[0] if env_type is None else self.meta_envs[env_type]
            elif isinstance(self.meta_envs, list):
                env = self.meta_envs[0]
            else:
                env = self.meta_envs
            
            alg = self.algorithm[0] if isinstance(self.algorithm, list) else self.algorithm
            
            episode_rewards = []
            episode_lengths = []
            
            for ep in range(n_episodes):
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    action = alg.select_action(obs)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            results = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths
            }
            print(f"Evaluation: mean_reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            return results
        
        def save(self, path):
            print(f"Saving to {path} (mock)")
        
        def load(self, path):
            print(f"Loading from {path} (mock)")
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseTrainer (SimpleTrainer implementation)")
    print("=" * 60)
    
    # Create components
    print("\n1. Creating components...")
    env = DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20)
    meta_policy = DummyMetaPolicy(action_dim=7)
    replay = SimpleReplay(capacity=10000)
    algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=replay)
    
    # Test 1: Basic trainer
    print("\n2. Testing basic trainer...")
    trainer = SimpleTrainer(
        meta_envs=env,
        algorithm=algorithm,
        reward_fn=None
    )
    print(f"   Trainer: {trainer}")
    
    # Test compute_reward without reward_fn
    print("\n3. Testing compute_reward without custom reward_fn...")
    state = MetaObs(state=np.random.randn(10).astype(np.float32))
    action = MetaAction(action=np.random.randn(7).astype(np.float32))
    next_state = MetaObs(state=np.random.randn(10).astype(np.float32))
    env_reward = 1.5
    computed_reward = trainer.compute_reward(state, action, next_state, env_reward, {})
    print(f"   Env reward: {env_reward}, Computed reward: {computed_reward}")
    assert computed_reward == env_reward
    
    # Test compute_reward with custom reward_fn
    print("\n4. Testing compute_reward with custom reward_fn...")
    trainer.set_reward_fn(ScaledReward(scale=2.0, offset=0.5))
    computed_reward = trainer.compute_reward(state, action, next_state, env_reward, {})
    expected = env_reward * 2.0 + 0.5
    print(f"   Env reward: {env_reward}, Computed reward: {computed_reward}, Expected: {expected}")
    assert abs(computed_reward - expected) < 1e-6
    
    # Test training
    print("\n5. Testing training...")
    trainer.train(total_steps=200, log_interval=50, update_interval=20, batch_size=16)
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Test evaluation
    print("\n6. Testing evaluation...")
    eval_results = trainer.evaluate(n_episodes=3)
    print(f"   Evaluation results: {eval_results}")
    
    # Test collect_rollout
    print("\n7. Testing collect_rollout...")
    rollout_stats = trainer.collect_rollout(n_steps=50)
    print(f"   Rollout stats: {rollout_stats}")
    
    # Test with multiple algorithms
    print("\n8. Testing with multiple algorithms...")
    algorithms = [
        DummyAlgorithm(meta_policy=DummyMetaPolicy(action_dim=7), replay=SimpleReplay(capacity=1000)),
        DummyAlgorithm(meta_policy=DummyMetaPolicy(action_dim=7), replay=SimpleReplay(capacity=1000))
    ]
    multi_trainer = SimpleTrainer(
        meta_envs=env,
        algorithm=algorithms,
        reward_fn=IdentityReward()
    )
    print(f"   Multi-algorithm trainer: {multi_trainer}")
    multi_trainer.train(total_steps=100, log_interval=50, update_interval=20, batch_size=8)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

