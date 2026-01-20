"""
Base classes for Reinforcement Learning in ILStudio.

This module provides:
1. RLPolicy - Base class for RL policy networks
2. RLTrainer - Base class for RL training loops (offline/online/hybrid)
3. RLConfig - Configuration dataclass for RL training

These base classes are designed to integrate with ILStudio's existing
infrastructure while supporting both offline and online RL algorithms.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from tqdm import tqdm

from .replay_buffer import ILReplayBuffer, BatchTransition
from .replay_buffer import RolloutReplayBuffer


class RLMode(Enum):
    """RL training mode."""
    OFFLINE = "offline"      # Pure offline RL (e.g., BC, IQL, CQL, TD3+BC)
    ONLINE = "online"        # Pure online RL (e.g., SAC, TD3, PPO)
    HYBRID = "hybrid"        # Offline pretraining + online finetuning


@dataclass
class RLConfig:
    """
    Configuration for RL training.
    
    This config extends ILStudio's training config with RL-specific parameters.
    """
    # Training mode
    mode: RLMode = RLMode.OFFLINE
    
    # Buffer configuration
    buffer_capacity: int = 1_000_000
    buffer_device: str = "cpu"  # Storage device for buffer
    
    # Sampling configuration
    batch_size: int = 256
    
    # Learning configuration
    learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005  # Target network update rate
    
    # Training steps
    total_steps: int = 1_000_000
    eval_freq: int = 5000
    save_freq: int = 10000
    log_freq: int = 1000
    
    # Online-specific
    warmup_steps: int = 1000  # Random actions before training
    env_steps_per_train: int = 1  # Environment steps per training step
    num_envs: int = 1  # Number of parallel environments for online training
    
    # Offline-specific
    offline_ratio: float = 0.5  # Ratio of offline data in hybrid mode
    
    # Environment configuration
    max_episode_steps: int = 1000
    num_eval_episodes: int = 10
    
    # Output
    output_dir: str = "ckpt/rl_output"
    seed: int = 0
    device: str = "cuda"
    
    # Normalization and transforms (extracted from dataset or set manually)
    action_normalizer: Optional[Any] = None
    state_normalizer: Optional[Any] = None
    transforms: Optional[Any] = None
    
    # Control space configuration
    ctrl_space: str = 'ee'
    ctrl_type: str = 'delta'
    chunk_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excludes non-serializable objects)."""
        return {
            'mode': self.mode.value,
            'buffer_capacity': self.buffer_capacity,
            'buffer_device': self.buffer_device,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'tau': self.tau,
            'total_steps': self.total_steps,
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'log_freq': self.log_freq,
            'warmup_steps': self.warmup_steps,
            'env_steps_per_train': self.env_steps_per_train,
            'num_envs': self.num_envs,
            'offline_ratio': self.offline_ratio,
            'max_episode_steps': self.max_episode_steps,
            'num_eval_episodes': self.num_eval_episodes,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
            'ctrl_space': self.ctrl_space,
            'ctrl_type': self.ctrl_type,
            'chunk_size': self.chunk_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLConfig":
        """Create config from dictionary."""
        if 'mode' in config_dict:
            config_dict['mode'] = RLMode(config_dict['mode'])
        return cls(**config_dict)


class RLPolicy(nn.Module, ABC):
    """
    Abstract base class for RL policies.
    
    Subclasses should implement:
    - forward(): Policy network forward pass
    - select_action(): Action selection (with optional exploration)
    - update(): Policy update step
    
    This class is compatible with ILStudio's policy loading system.
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__()
        self.config = config
        self._training_mode = True
    
    @abstractmethod
    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: Dictionary of state tensors (e.g., {'image': ..., 'state': ...})
            
        Returns:
            Action tensor or action distribution parameters
        """
        pass
    
    @abstractmethod
    def select_action(
        self,
        state: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action given state.
        
        Args:
            state: Dictionary of state tensors
            deterministic: If True, return deterministic action (for evaluation)
            
        Returns:
            Action tensor
        """
        pass
    
    @abstractmethod
    def update(
        self,
        batch: BatchTransition,
    ) -> Dict[str, float]:
        """
        Update policy from a batch of transitions.
        
        Args:
            batch: BatchTransition containing (s, a, r, s', done)
            
        Returns:
            Dictionary of training metrics (losses, etc.)
        """
        pass
    
    def reset(self):
        """Reset policy state (e.g., for recurrent policies)."""
        pass
    
    def set_training_mode(self, mode: bool):
        """Set training/evaluation mode."""
        self._training_mode = mode
        self.train(mode)
    
    def save(self, save_dir: str):
        """Save policy checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "policy.pt"))
        if self.config is not None:
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            with open(os.path.join(save_dir, "config.json"), 'w') as f:
                json.dump(config_dict, f, indent=2)
        logger.info(f"Policy saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, device: str = "cuda"):
        """Load policy from checkpoint."""
        # Load config
        config_path = os.path.join(load_dir, "config.json")
        config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create policy instance (subclass should handle config parsing)
        # This is a simplified version - subclasses should override
        policy = cls(config)
        
        # Load weights
        weights_path = os.path.join(load_dir, "policy.pt")
        if os.path.exists(weights_path):
            policy.load_state_dict(torch.load(weights_path, map_location=device))
        
        policy.to(device)
        logger.info(f"Policy loaded from {load_dir}")
        return policy


class RLTrainer(ABC):
    """
    Base trainer for Reinforcement Learning.
    
    Supports three training modes:
    1. OFFLINE: Train purely from offline dataset (imitation learning, offline RL)
    2. ONLINE: Train through environment interaction
    3. HYBRID: Offline pretraining followed by online finetuning
    
    Designed to integrate with ILStudio's train.py workflow.
    """
    
    def __init__(
        self,
        config: RLConfig,
        policy: RLPolicy,
        replay_buffer: Optional[ILReplayBuffer] = None,
        env=None,  # Vectorized environment for online training
        eval_env=None,  # Environment for evaluation
    ):
        self.config = config
        self.policy = policy
        self.env = env
        self.eval_env = eval_env or env
        
        # Initialize or use provided replay buffer
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = ILReplayBuffer(
                capacity=config.buffer_capacity,
                chunk_size=config.chunk_size,
                action_normalizer=config.action_normalizer,
                state_normalizer=config.state_normalizer,
                transforms=config.transforms,
                ctrl_space=config.ctrl_space,
                ctrl_type=config.ctrl_type,
                device=config.device,
                storage_device=config.buffer_device,
            )
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_eval_return = -float('inf')
        
        # Logging
        self.train_metrics = {}
        self.eval_metrics = {}
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_offline_data(
        self, 
        dataset,
        action_normalizer=None,
        state_normalizer=None,
        transforms=None,
    ):
        """
        Load offline dataset into replay buffer.
        
        This method integrates with ILStudio's data loading:
        ```python
        data_dict = load_data(args, task_config)
        trainer.load_offline_data(data_dict['train'])
        ```
        
        The method will automatically extract normalizers and transforms from
        the dataset wrapper chain if not explicitly provided.
        
        Args:
            dataset: ILStudio dataset from load_data() (may be wrapped with normalizers/transforms)
            action_normalizer: Optional action normalizer (extracted from dataset if None)
            state_normalizer: Optional state normalizer (extracted from dataset if None)
            transforms: Optional transforms pipeline (extracted from dataset if None)
        """
        logger.info("Loading offline data into replay buffer...")
        
        # Clear existing buffer if needed
        if self.replay_buffer.size > 0:
            logger.warning("Clearing existing buffer data before loading offline data")
            self.replay_buffer.clear()
        
        # Load data - normalizers/transforms will be extracted from dataset wrapper chain
        self.replay_buffer = ILReplayBuffer.from_ilstudio_dataset(
            raw_dataset=dataset,
            capacity=self.config.buffer_capacity,
            chunk_size=self.config.chunk_size,
            action_normalizer=action_normalizer or self.config.action_normalizer,
            state_normalizer=state_normalizer or self.config.state_normalizer,
            transforms=transforms or self.config.transforms,
            ctrl_space=self.config.ctrl_space,
            ctrl_type=self.config.ctrl_type,
            device=self.config.device,
            storage_device=self.config.buffer_device,
            store_raw=True,
        )
        
        stats = self.replay_buffer.get_statistics()
        logger.info(f"Loaded {stats['size']} transitions from offline data")
        logger.info(f"Buffer statistics: {stats}")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting RL training in {self.config.mode.value} mode")
        logger.info(f"Total steps: {self.config.total_steps}")
        
        if self.config.mode == RLMode.OFFLINE:
            self._train_offline()
        elif self.config.mode == RLMode.ONLINE:
            self._train_online()
        elif self.config.mode == RLMode.HYBRID:
            self._train_hybrid()
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
        
        # Final save
        self.save_checkpoint(os.path.join(self.config.output_dir, "final"))
        logger.info("Training completed!")
    
    def _train_offline(self):
        """
        Offline training loop.
        
        Trains purely from data in replay buffer without environment interaction.
        Used for imitation learning, offline RL (IQL, CQL, etc.)
        """
        if self.replay_buffer.size == 0:
            raise RuntimeError("Replay buffer is empty. Load offline data first with load_offline_data()")
        
        logger.info(f"Starting offline training with {self.replay_buffer.size} transitions")
        
        # Create batch iterator
        batch_iterator = self.replay_buffer.get_iterator(
            batch_size=self.config.batch_size,
            async_prefetch=True,
        )
        
        pbar = tqdm(total=self.config.total_steps, desc="Offline Training")
        pbar.update(self.global_step)
        
        while self.global_step < self.config.total_steps:
            # Sample batch and update
            batch = next(batch_iterator)
            metrics = self.policy.update(batch)
            
            # Accumulate metrics
            for k, v in metrics.items():
                if k not in self.train_metrics:
                    self.train_metrics[k] = []
                self.train_metrics[k].append(v)
            
            self.global_step += 1
            pbar.update(1)
            
            # Logging
            if self.global_step % self.config.log_freq == 0:
                self._log_metrics()
            
            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                self._evaluate()
            
            # Checkpointing
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                )
        
        pbar.close()
    
    def _train_online(self):
        """
        Online training loop with parallel environment support.
        
        Trains through environment interaction, adding new transitions to buffer.
        Supports vectorized environments for faster data collection.
        Used for online RL (SAC, TD3, PPO, etc.)
        """
        if self.env is None:
            raise RuntimeError("Environment not provided for online training")
        
        num_envs = self.config.num_envs
        logger.info(f"Starting online training with {num_envs} parallel environment(s)")
        
        # Use RolloutReplayBuffer for parallel environments if num_envs > 1
        if num_envs > 1 and not isinstance(self.replay_buffer, RolloutReplayBuffer):
            logger.info("Switching to RolloutReplayBuffer for parallel environment support")
            self.replay_buffer = RolloutReplayBuffer(
                capacity=self.config.buffer_capacity,
                num_envs=num_envs,
                chunk_size=self.config.chunk_size,
                ctrl_space=self.config.ctrl_space,
                ctrl_type=self.config.ctrl_type,
                device=self.config.device,
                storage_device=self.config.buffer_device,
            )
        
        # Reset environments
        obs = self._reset_env()
        obs_dict = self._obs_to_dict(obs)
        
        # Per-environment tracking
        episode_returns = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        env_done = np.zeros(num_envs, dtype=bool)  # Track which envs have finished current episode
        
        pbar = tqdm(total=self.config.total_steps, desc="Online Training")
        pbar.update(self.global_step)
        
        timestep = 0  # For action chunking policies
        
        while self.global_step < self.config.total_steps:
            # Select action
            if self.global_step < self.config.warmup_steps:
                # Random action during warmup
                action_batch = self._sample_random_action_batch(num_envs)
            else:
                with torch.no_grad():
                    action_batch = self._select_action_batch(obs, timestep, deterministic=False)
            
            # Environment step
            next_obs, reward, done, info = self.env.step(action_batch)
            next_obs_dict = self._obs_to_dict(next_obs)
            reward = np.array(reward)
            done = np.array(done)
            
            # Add to buffer (handle parallel envs)
            if isinstance(self.replay_buffer, RolloutReplayBuffer):
                self.replay_buffer.add_from_parallel_envs(
                    obs_batch=obs_dict,
                    action_batch=action_batch,
                    reward_batch=reward,
                    next_obs_batch=next_obs_dict,
                    done_batch=done,
                    info_batch=info,
                    skip_mask=env_done,  # Skip envs that already finished
                )
            else:
                # Single env fallback
                self.replay_buffer.add_from_env_step(
                    obs=obs,
                    action=action_batch[0] if action_batch.ndim > 1 else action_batch,
                    reward=float(reward[0]) if hasattr(reward, '__len__') else reward,
                    next_obs=next_obs,
                    done=bool(done[0]) if hasattr(done, '__len__') else done,
                    truncated=False,
                    info=info[0] if isinstance(info, list) else info,
                )
            
            # Update per-env statistics (only for active envs)
            active_mask = ~env_done
            episode_returns[active_mask] += reward[active_mask]
            episode_lengths[active_mask] += 1
            
            # Track newly done environments
            newly_done = done & ~env_done
            
            # Log completed episodes
            for idx in range(num_envs):
                if newly_done[idx]:
                    self.episode_count += 1
                    self.train_metrics.setdefault('episode_return', []).append(float(episode_returns[idx]))
                    self.train_metrics.setdefault('episode_length', []).append(int(episode_lengths[idx]))
                    
                    # Check for success in info
                    if isinstance(info, list) and len(info) > idx:
                        env_info = info[idx]
                        if isinstance(env_info, dict) and 'success' in env_info:
                            self.train_metrics.setdefault('episode_success', []).append(float(env_info['success']))
            
            # Mark environments as done AFTER recording their final transition
            env_done = env_done | done
            
            # Reset statistics for done environments (they auto-reset in vectorized envs)
            episode_returns[done] = 0
            episode_lengths[done] = 0
            env_done[done] = False  # Ready for new episode
            
            # Move to next state
            obs = next_obs
            obs_dict = next_obs_dict
            timestep += 1
            
            # Policy update
            if self.global_step >= self.config.warmup_steps and self.replay_buffer.size >= self.config.batch_size:
                for _ in range(self.config.env_steps_per_train):
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    metrics = self.policy.update(batch)
                    
                    for k, v in metrics.items():
                        self.train_metrics.setdefault(k, []).append(v)
            
            self.global_step += num_envs  # Count all env steps
            pbar.update(num_envs)
            
            # Logging
            if self.global_step % self.config.log_freq == 0:
                self._log_metrics()
            
            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                self._evaluate()
            
            # Checkpointing
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                )
        
        pbar.close()
    
    def _train_hybrid(self):
        """
        Hybrid training: offline pretraining + online finetuning.
        
        First trains offline, then continues with online training.
        During online training, samples from both offline and online data.
        """
        logger.info("Starting hybrid training (offline pretraining + online finetuning)")
        
        # Phase 1: Offline pretraining
        offline_steps = int(self.config.total_steps * 0.5)  # 50% offline
        logger.info(f"Phase 1: Offline pretraining for {offline_steps} steps")
        
        original_total = self.config.total_steps
        self.config.total_steps = offline_steps
        self._train_offline()
        self.config.total_steps = original_total
        
        # Phase 2: Online finetuning
        logger.info(f"Phase 2: Online finetuning for remaining steps")
        if self.env is not None:
            self._train_online()
        else:
            logger.warning("No environment provided, skipping online phase")
    
    def _evaluate(self):
        """
        Evaluate current policy.
        
        Runs evaluation episodes and logs metrics.
        """
        if self.eval_env is None:
            logger.warning("No evaluation environment, skipping evaluation")
            return
        
        self.policy.set_training_mode(False)
        
        eval_returns = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(self.config.num_eval_episodes):
            obs, info = self._reset_env(self.eval_env)
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.config.max_episode_steps:
                with torch.no_grad():
                    state = self._obs_to_state_tensor(obs)
                    action = self.policy.select_action(state, deterministic=True)
                    action = action.cpu().numpy()
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
            if 'success' in info:
                eval_successes.append(float(info['success']))
        
        # Log evaluation metrics
        mean_return = np.mean(eval_returns)
        self.eval_metrics['eval_return'] = mean_return
        self.eval_metrics['eval_length'] = np.mean(eval_lengths)
        if eval_successes:
            self.eval_metrics['eval_success_rate'] = np.mean(eval_successes)
        
        logger.info(f"Evaluation at step {self.global_step}: "
                   f"mean_return={mean_return:.2f}, "
                   f"mean_length={np.mean(eval_lengths):.1f}")
        
        # Save best model
        if mean_return > self.best_eval_return:
            self.best_eval_return = mean_return
            self.save_checkpoint(os.path.join(self.config.output_dir, "best"))
            logger.info(f"New best model saved with return {mean_return:.2f}")
        
        self.policy.set_training_mode(True)
    
    def _log_metrics(self):
        """Log accumulated training metrics."""
        log_str = f"Step {self.global_step}"
        
        for k, v in self.train_metrics.items():
            if v:
                mean_v = np.mean(v[-self.config.log_freq:])
                log_str += f" | {k}: {mean_v:.4f}"
        
        logger.info(log_str)
    
    def _reset_env(self, env=None):
        """
        Reset environment(s).
        
        Handles both single and vectorized environments.
        Returns observation (and optionally info for gymnasium envs).
        """
        env = env or self.env
        result = env.reset()
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            # Gymnasium format: (obs, info)
            return result[0]
        else:
            # Old gym format or vectorized env: just obs
            return result
    
    def _sample_random_action(self):
        """Sample random action from action space."""
        return self.env.action_space.sample()
    
    def _sample_random_action_batch(self, num_envs: int) -> np.ndarray:
        """Sample random actions for multiple environments."""
        actions = []
        for _ in range(num_envs):
            actions.append(self.env.action_space.sample())
        return np.array(actions)
    
    def _obs_to_dict(self, obs) -> Dict[str, np.ndarray]:
        """
        Convert observation to dict format for replay buffer.
        
        Handles various observation formats:
        - Dict observations
        - Dataclass observations (MetaObs)
        - Batched observations from vectorized envs
        """
        from dataclasses import asdict
        
        if isinstance(obs, dict):
            return obs
        elif hasattr(obs, '__dataclass_fields__'):
            return asdict(obs)
        elif isinstance(obs, (list, tuple)):
            # Batched observations - convert first one to get structure
            if len(obs) > 0:
                first = obs[0]
                if isinstance(first, dict):
                    # Stack dict values
                    result = {}
                    for key in first.keys():
                        values = [o[key] for o in obs]
                        if isinstance(values[0], np.ndarray):
                            result[key] = np.stack(values)
                        else:
                            result[key] = values
                    return result
                elif hasattr(first, '__dataclass_fields__'):
                    # Stack dataclass fields
                    result = {}
                    for key in first.__dataclass_fields__.keys():
                        values = [getattr(o, key) for o in obs]
                        if isinstance(values[0], np.ndarray):
                            result[key] = np.stack(values)
                        else:
                            result[key] = values
                    return result
            return {}
        else:
            raise TypeError(f"Unknown observation type: {type(obs)}")
    
    def _select_action_batch(
        self, 
        obs, 
        timestep: int = 0, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select actions for batch of observations.
        
        Args:
            obs: Observations (can be batched or list)
            timestep: Current timestep (for action chunking policies)
            deterministic: If True, return deterministic actions
            
        Returns:
            Action batch as numpy array (num_envs, action_dim)
        """
        # Convert obs to tensor format
        obs_dict = self._obs_to_dict(obs)
        
        # Try using policy's select_action if it supports batched input
        if hasattr(self.policy, 'select_action'):
            # Create batched state tensor
            state = {}
            for key, val in obs_dict.items():
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float().to(self.config.device)
                elif isinstance(val, torch.Tensor):
                    val = val.float().to(self.config.device)
                else:
                    continue
                state[key] = val
            
            action = self.policy.select_action(state, deterministic=deterministic)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            elif hasattr(action, 'action'):  # MetaAction
                action = action.action
            
            return action
        else:
            # Fallback: forward pass
            with torch.no_grad():
                state = self._obs_to_state_tensor(obs_dict)
                action = self.policy(state)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
            return action
    
    def _obs_to_state_tensor(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to state tensor dict on device."""
        state = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            elif not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            # Add batch dim if needed
            if val.dim() == 1 or (val.dim() > 1 and val.shape[0] != 1):
                val = val.unsqueeze(0)
            state[key] = val.to(self.config.device)
        return state
    
    def save_checkpoint(self, save_dir: str):
        """Save training checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save policy
        self.policy.save(save_dir)
        
        # Save trainer state
        trainer_state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'best_eval_return': self.best_eval_return,
            'train_metrics': {k: v[-1000:] for k, v in self.train_metrics.items()},  # Last 1000
            'eval_metrics': self.eval_metrics,
        }
        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
        
        # Save config
        with open(os.path.join(save_dir, "rl_config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_dir}")
    
    def load_checkpoint(self, load_dir: str):
        """Load training checkpoint."""
        # Load policy
        policy_path = os.path.join(load_dir, "policy.pt")
        if os.path.exists(policy_path):
            self.policy.load_state_dict(
                torch.load(policy_path, map_location=self.config.device)
            )
        
        # Load trainer state
        state_path = os.path.join(load_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            self.global_step = state.get('global_step', 0)
            self.episode_count = state.get('episode_count', 0)
            self.best_eval_return = state.get('best_eval_return', -float('inf'))
            self.train_metrics = state.get('train_metrics', {})
            self.eval_metrics = state.get('eval_metrics', {})
        
        logger.info(f"Checkpoint loaded from {load_dir}, step={self.global_step}")
    
    def is_world_process_zero(self) -> bool:
        """Check if this is the main process (for distributed training)."""
        # Simple implementation - can be extended for distributed
        return True
    
    def save_model(self, output_dir: str):
        """Save final model (compatible with ILStudio's trainer interface)."""
        self.policy.save(output_dir)
    
    def save_state(self):
        """Save training state (compatible with ILStudio's trainer interface)."""
        self.save_checkpoint(
            os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        )


class OfflineRLTrainer(RLTrainer):
    """
    Convenience trainer for offline RL.
    
    Pre-configured for offline training mode.
    """
    
    def __init__(self, config: RLConfig, policy: RLPolicy, **kwargs):
        config.mode = RLMode.OFFLINE
        super().__init__(config=config, policy=policy, **kwargs)


class OnlineRLTrainer(RLTrainer):
    """
    Convenience trainer for online RL.
    
    Pre-configured for online training mode.
    Requires environment to be provided.
    """
    
    def __init__(self, config: RLConfig, policy: RLPolicy, env, **kwargs):
        config.mode = RLMode.ONLINE
        super().__init__(config=config, policy=policy, env=env, **kwargs)


class HybridRLTrainer(RLTrainer):
    """
    Convenience trainer for hybrid (offline + online) RL.
    
    Pre-configured for hybrid training mode.
    """
    
    def __init__(self, config: RLConfig, policy: RLPolicy, env=None, **kwargs):
        config.mode = RLMode.HYBRID
        super().__init__(config=config, policy=policy, env=env, **kwargs)











