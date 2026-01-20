"""
IQL Trainer for ILStudio.

This trainer implements offline reinforcement learning using IQL.
It uses ILStudio's ILReplayBuffer for unified data handling.
"""

import os
import time
from typing import Dict, Optional, Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from policy.rl.base import RLTrainer, RLConfig, RLMode
from policy.rl.replay_buffer import ILReplayBuffer
from policy.rl.offline.iql.modeling import IQLAgent, IQLConfig


class IQLTrainer(RLTrainer):
    """
    Trainer for IQL (Implicit Q-Learning).
    
    IQL is an offline RL algorithm that learns from a fixed dataset
    without environment interaction during training.
    
    Uses ILStudio's ILReplayBuffer for:
    - Unified interface with other ILStudio components
    - Efficient sampling
    - Support for various observation formats (state, image, etc.)
    """
    
    def __init__(
        self,
        config: RLConfig,
        policy: IQLAgent,
        replay_buffer: ILReplayBuffer,
        eval_env=None,
        **kwargs
    ):
        """
        Initialize IQL trainer.
        
        Args:
            config: RL training configuration
            policy: IQL agent
            replay_buffer: ILStudio ILReplayBuffer containing offline dataset
            eval_env: Optional evaluation environment
        """
        # For offline training, env is not needed
        super().__init__(config=config, policy=policy, env=None, eval_env=eval_env)
        
        self.iql_agent = policy
        self.replay_buffer = replay_buffer
        self.dataset_size = replay_buffer.size
        
        logger.info(f"IQLTrainer initialized with ReplayBuffer: {self.dataset_size} transitions")
        
        # Log buffer info
        if hasattr(replay_buffer, 'obs_storage'):
            for key in replay_buffer.obs_storage:
                if key != 'raw_lang':
                    shape = replay_buffer.obs_storage[key].shape
                    logger.info(f"  obs/{key}: shape={shape}")
        if hasattr(replay_buffer, 'actions'):
            logger.info(f"  actions: shape={replay_buffer.actions.shape}")
    
    def _sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch from the ReplayBuffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dict with keys: observations, actions, next_observations, rewards, terminals
        """
        device = torch.device(self.config.device)
        
        # Use buffer's sample method (returns BatchTransition dict)
        batch = self.replay_buffer.sample(batch_size)
        
        # BatchTransition is a TypedDict, access via ['key']
        obs_dict = batch['obs']
        next_obs_dict = batch['next_obs']
        
        # Extract state observations
        if 'state' in obs_dict:
            observations = obs_dict['state']
            next_observations = next_obs_dict['state']
        elif 'observation' in obs_dict:
            observations = obs_dict['observation']
            next_observations = next_obs_dict['observation']
        else:
            # Fallback: use first available key (excluding special keys)
            available_keys = [k for k in obs_dict.keys() 
                            if k not in ('raw_lang',) and isinstance(obs_dict[k], torch.Tensor)]
            if available_keys:
                key = available_keys[0]
                observations = obs_dict[key]
                next_observations = next_obs_dict[key]
                
                # Flatten if needed (e.g., image observations)
                if observations.dim() > 2:
                    observations = observations.view(observations.size(0), -1)
                    next_observations = next_observations.view(next_observations.size(0), -1)
            else:
                raise ValueError("No valid observation key found in replay buffer")
        
        # Handle action shape (may have chunk dimension)
        actions = batch['action']
        if actions.dim() == 3:
            # (B, chunk_size, action_dim) -> (B, action_dim)
            actions = actions[:, 0, :]
        
        return {
            'observations': observations.to(device).float(),
            'actions': actions.to(device).float(),
            'next_observations': next_observations.to(device).float(),
            'rewards': batch['reward'].to(device).float(),
            'terminals': batch['done'].to(device).float(),
        }
    
    def _train_offline(self):
        """
        Offline training loop for IQL.
        
        Trains on the fixed dataset without environment interaction.
        """
        logger.info(f"Starting IQL offline training")
        logger.info(f"Total steps: {self.config.total_steps}")
        logger.info(f"Dataset size: {self.dataset_size}")
        logger.info(f"Batch size: {self.config.batch_size}")
        
        pbar = tqdm(total=self.config.total_steps, desc="IQL Training")
        pbar.update(self.global_step)
        
        # Time tracking
        time_stats = {
            'sample': [],
            'update': [],
            'total': [],
        }
        
        while self.global_step < self.config.total_steps:
            step_start = time.perf_counter()
            
            # Sample batch
            sample_start = time.perf_counter()
            batch = self._sample_batch(self.config.batch_size)
            time_stats['sample'].append(time.perf_counter() - sample_start)
            
            # Update
            update_start = time.perf_counter()
            metrics = self.iql_agent.update(
                observations=batch['observations'],
                actions=batch['actions'],
                next_observations=batch['next_observations'],
                rewards=batch['rewards'],
                terminals=batch['terminals'],
            )
            time_stats['update'].append(time.perf_counter() - update_start)
            
            # Record metrics
            for k, v in metrics.items():
                self.train_metrics.setdefault(k, []).append(v)
            
            time_stats['total'].append(time.perf_counter() - step_start)
            
            self.global_step += 1
            pbar.update(1)
            
            # Logging
            if self.global_step % self.config.log_freq == 0:
                self._log_metrics()
                self._log_timing_stats(time_stats)
                time_stats = {k: [] for k in time_stats}
            
            # Evaluation
            if self.config.eval_freq > 0 and self.global_step % self.config.eval_freq == 0:
                self._evaluate_iql()
            
            # Checkpointing
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                )
        
        pbar.close()
        
        # Final save
        self.save_checkpoint(os.path.join(self.config.output_dir, "final"))
    
    def _evaluate_iql(self):
        """Evaluate IQL agent in the environment."""
        if self.eval_env is None:
            logger.warning("No evaluation environment provided, skipping evaluation")
            return
        
        self.iql_agent.eval()
        
        eval_returns = []
        eval_lengths = []
        
        for ep in range(self.config.num_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done and episode_length < self.config.max_episode_steps:
                # Get observation array
                if hasattr(obs, 'state'):
                    obs_array = obs.state
                elif isinstance(obs, dict):
                    obs_array = obs.get('state', obs.get('observation'))
                elif isinstance(obs, np.ndarray):
                    obs_array = obs
                else:
                    obs_array = np.array(obs)
                
                action = self.iql_agent.act(obs_array, deterministic=True)
                
                result = self.eval_env.step(action)
                if len(result) == 4:
                    obs, reward, done, info = result
                else:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
            
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
        
        self.iql_agent.train()
        
        # Log evaluation results
        mean_return = np.mean(eval_returns)
        std_return = np.std(eval_returns)
        mean_length = np.mean(eval_lengths)
        
        logger.info(f"Evaluation at step {self.global_step}:")
        logger.info(f"  Return: {mean_return:.2f} ± {std_return:.2f}")
        logger.info(f"  Length: {mean_length:.1f}")
        
        self.train_metrics.setdefault('eval_return', []).append(mean_return)
        self.train_metrics.setdefault('eval_length', []).append(mean_length)
    
    def _log_timing_stats(self, time_stats: Dict[str, list]):
        """Log timing statistics."""
        if not any(time_stats.values()):
            return
        
        log_parts = []
        for key, times in time_stats.items():
            if times:
                mean_ms = np.mean(times) * 1000
                log_parts.append(f"{key}={mean_ms:.2f}ms")
        
        if log_parts:
            logger.info(f"Timing: {' '.join(log_parts)}")
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save agent
        agent_path = os.path.join(path, "agent.pt")
        self.iql_agent.save(agent_path)
        
        # Save trainer state
        trainer_state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        }
        torch.save(trainer_state, os.path.join(path, "trainer_state.pt"))
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # Load agent
        agent_path = os.path.join(path, "agent.pt")
        if os.path.exists(agent_path):
            self.iql_agent.load(agent_path)
        
        # Load trainer state
        trainer_state_path = os.path.join(path, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            state = torch.load(trainer_state_path, map_location=self.config.device)
            self.global_step = state.get('global_step', 0)
            self.episode_count = state.get('episode_count', 0)
        
        logger.info(f"Checkpoint loaded from {path}, step={self.global_step}")
    
    def train(self):
        """Main training entry point."""
        logger.info("Starting IQL training in offline mode")
        logger.info(f"Total steps: {self.config.total_steps}")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self._train_offline()
        
        logger.info("Training completed!")
