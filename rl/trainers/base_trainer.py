"""
Base Trainer Class

This module defines the base class for all RL trainers in the framework.

Design Philosophy:
- Coordinate environment, policy, and algorithm for executing training loop
- Support single algorithm and multiple algorithms training scenarios
- Support custom reward functions (applied during training, not data collection)
- Support evaluation during training
- Support vectorized environments (SequentialVectorEnv, SubprocVectorEnv, etc.)
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List, Callable
from abc import ABC, abstractmethod

# Type hints for Meta classes
from benchmark.base import MetaObs, MetaAction

# Vector environment protocol
from rl.envs import VectorEnvProtocol, EnvsType


class BaseTrainer(ABC):
    """
    Base class for RL trainers.
    
    This class defines the interface for all trainers in the RL framework.
    Trainers coordinate the training loop by managing:
    - Vectorized environments (envs)
    - Algorithms (with their policies and replay buffers)
    - Collectors (for data collection)
    - Reward functions (applied during training)
    
    Attributes:
        envs: Vectorized environment(s) for training
        algorithm: Algorithm(s) for training
        collector: Data collector(s)
        reward_fn: Optional custom reward function
    """
    
    def __init__(
        self,
        envs: Union[VectorEnvProtocol, Dict[str, VectorEnvProtocol]],
        algorithm: Union['BaseAlgorithm', List['BaseAlgorithm']],
        collector: Optional[Union['BaseCollector', List['BaseCollector']]] = None,
        reward_fn: Optional[Union['BaseReward', Callable]] = None,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            envs: Vectorized environment(s), supports:
                - VectorEnvProtocol: Single vectorized environment
                  (SequentialVectorEnv, SubprocVectorEnv, DummyVectorEnv, etc.)
                - Dict[str, VectorEnvProtocol]: Multi-environment dict for different env types
                  e.g., {'sim': sim_vec_env, 'real': real_vec_env}
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
        self.envs = envs
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self._kwargs = kwargs
        
        # Normalize environment storage: always use dict internally
        if isinstance(envs, dict):
            self._envs_dict: Dict[str, VectorEnvProtocol] = envs
            self._is_multi_env = True
        else:
            self._envs_dict = {'default': envs}
            self._is_multi_env = False
        
        # Handle collector initialization
        self.collector = collector
        # Note: Actual collector creation should be done in subclasses
        # since it may require specific collector types
    
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
        env_info = f"{self.get_total_env_num()} envs" if self._is_multi_env else f"{self.env_num} envs"
        return f"{self.__class__.__name__}(envs={env_info}, algorithm={algo_info}, collector={collector_info}, reward_fn={reward_info})"


if __name__ == '__main__':
    """
    Test code for BaseTrainer class.
    
    Since BaseTrainer is abstract, we create a simple concrete implementation for testing.
    Tests now use vectorized environments (VectorEnvProtocol).
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction, MetaEnv, MetaPolicy
    from rl.buffer.base_replay import BaseReplay
    from rl.base import BaseAlgorithm
    from rl.rewards.base_reward import BaseReward, IdentityReward, ScaledReward
    from rl.collectors.base_collector import BaseCollector
    from rl.envs import VectorEnvProtocol
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
    
    # Simple collector for testing (works with VectorEnv)
    class DummyCollector(BaseCollector):
        def __init__(self, envs, algorithm, **kwargs):
            super().__init__(envs, algorithm, **kwargs)
            self.vec_env = envs
            self._last_obs = None
        
        def reset(self, **kwargs):
            self._last_obs = self.vec_env.reset()
        
        def collect(self, n_steps, env_type=None):
            if self._last_obs is None:
                self.reset()
            
            stats = {'episode_rewards': [], 'episode_lengths': [], 'total_steps': 0}
            episode_rewards = np.zeros(len(self.vec_env))
            episode_lengths = np.zeros(len(self.vec_env), dtype=int)
            
            for step in range(n_steps):
                # Get actions for all environments
                actions = []
                for obs in self._last_obs:
                    action = self.algorithm.select_action(obs)
                    actions.append(action)
                
                # Step all environments
                new_obs, rewards, dones, infos = self.vec_env.step(actions)
                
                # Record transitions and update stats
                for i in range(len(self.vec_env)):
                    kwargs_trans = {'env_type': env_type} if env_type else {}
                    self.algorithm.record_transition(
                        state=self._last_obs[i], action=actions[i], reward=rewards[i],
                        next_state=new_obs[i], done=dones[i], info=infos[i], **kwargs_trans
                    )
                    
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    stats['total_steps'] += 1
                    
                    if dones[i]:
                        stats['episode_rewards'].append(episode_rewards[i])
                        stats['episode_lengths'].append(episode_lengths[i])
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        # Reset this env
                        new_obs[i] = self.vec_env.reset(id=i)
                
                self._last_obs = new_obs
            
            return stats
    
    # Simple trainer implementation for testing
    class SimpleTrainer(BaseTrainer):
        """Simple trainer for testing with vectorized environments."""
        
        def __init__(
            self,
            envs,
            algorithm,
            collector=None,
            reward_fn=None,
            **kwargs
        ):
            super().__init__(envs, algorithm, collector, reward_fn, **kwargs)
            
            # Create default collector if not provided
            if self.collector is None:
                default_env = self.get_env()
                if isinstance(algorithm, list):
                    self.collector = [
                        DummyCollector(envs=default_env, algorithm=alg)
                        for alg in algorithm
                    ]
                else:
                    self.collector = DummyCollector(
                        envs=default_env,
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
            
            # Get eval env
            vec_env = self.get_env(env_type) if env_type else self.get_env()
            alg = self.algorithm[0] if isinstance(self.algorithm, list) else self.algorithm
            
            episode_rewards = []
            episode_lengths = []
            
            for ep in range(n_episodes):
                obs = vec_env.reset(id=0)  # Reset first env only for single episode eval
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    action = alg.select_action(obs)
                    obs, reward, done, info = vec_env.step(action, id=0)
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
    print("Testing BaseTrainer with Vectorized Environments")
    print("=" * 60)
    
    # Create components
    print("\n1. Creating components...")
    # Create vectorized environment with 4 parallel envs
    env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20) for _ in range(4)]
    vec_env = DummyVectorEnv(env_fns)
    print(f"   Created DummyVectorEnv with {vec_env.env_num} parallel environments")
    
    meta_policy = DummyMetaPolicy(action_dim=7)
    replay = SimpleReplay(capacity=10000)
    algorithm = DummyAlgorithm(meta_policy=meta_policy, replay=replay)
    
    # Test 1: Basic trainer with vectorized env
    print("\n2. Testing basic trainer with vectorized env...")
    trainer = SimpleTrainer(
        envs=vec_env,
        algorithm=algorithm,
        reward_fn=None
    )
    print(f"   Trainer: {trainer}")
    print(f"   env_num: {trainer.env_num}")
    print(f"   total_env_num: {trainer.get_total_env_num()}")
    print(f"   env_types: {trainer.get_env_types()}")
    
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
    print("\n5. Testing training with vectorized env...")
    trainer.train(total_steps=400, log_interval=100, update_interval=20, batch_size=16)
    print(f"   Replay buffer size: {len(algorithm.replay)}")
    
    # Test evaluation
    print("\n6. Testing evaluation...")
    eval_results = trainer.evaluate(n_episodes=3)
    print(f"   Evaluation results: mean_reward={eval_results['mean_reward']:.2f}")
    
    # Test collect_rollout
    print("\n7. Testing collect_rollout...")
    rollout_stats = trainer.collect_rollout(n_steps=50)
    print(f"   Rollout stats: total_steps={rollout_stats['total_steps']}")
    
    # Test with multiple environment types (dict)
    print("\n8. Testing with multiple environment types (dict)...")
    sim_env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=20) for _ in range(4)]
    real_env_fns = [lambda: DummyMetaEnv(state_dim=10, action_dim=7, max_steps=10) for _ in range(2)]
    
    multi_envs = {
        'sim': DummyVectorEnv(sim_env_fns),
        'real': DummyVectorEnv(real_env_fns),
    }
    
    multi_env_trainer = SimpleTrainer(
        envs=multi_envs,
        algorithm=DummyAlgorithm(meta_policy=DummyMetaPolicy(action_dim=7), replay=SimpleReplay(capacity=1000)),
        reward_fn=IdentityReward()
    )
    print(f"   Multi-env trainer: {multi_env_trainer}")
    print(f"   env_types: {multi_env_trainer.get_env_types()}")
    print(f"   total_env_num: {multi_env_trainer.get_total_env_num()}")
    print(f"   sim env_num: {len(multi_env_trainer.get_env('sim'))}")
    print(f"   real env_num: {len(multi_env_trainer.get_env('real'))}")
    
    # Test with multiple algorithms
    print("\n9. Testing with multiple algorithms...")
    algorithms = [
        DummyAlgorithm(meta_policy=DummyMetaPolicy(action_dim=7), replay=SimpleReplay(capacity=1000)),
        DummyAlgorithm(meta_policy=DummyMetaPolicy(action_dim=7), replay=SimpleReplay(capacity=1000))
    ]
    multi_algo_trainer = SimpleTrainer(
        envs=vec_env,
        algorithm=algorithms,
        reward_fn=IdentityReward()
    )
    print(f"   Multi-algorithm trainer: {multi_algo_trainer}")
    multi_algo_trainer.train(total_steps=100, log_interval=50, update_interval=20, batch_size=8)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

