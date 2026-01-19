"""
DrQ Trainer for ILStudio.

This module provides the DrQ trainer that inherits from RLTrainer,
integrating with ILStudio's RL infrastructure.

DrQ uses RolloutReplayBuffer for data storage and applies random shift
augmentation during the update step.
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, Any, Optional
from loguru import logger

from policy.rl.base import RLTrainer, RLConfig, RLMode, RLPolicy
from policy.rl.rollout_buffer import RolloutReplayBuffer
from .modeling import DrQAgent, DrQConfig
from .data_utils import create_augmentation, apply_drq_augmentation


class DrQTrainer(RLTrainer):
    """
    Trainer for DrQ algorithm.
    
    Inherits from RLTrainer and specializes for DrQ:
    1. Uses RolloutReplayBuffer for data collection
    2. Applies random shift augmentation during update
    3. Handles image-based observations
    
    Example:
        >>> config = RLConfig(mode=RLMode.ONLINE, ...)
        >>> agent = DrQAgent(drq_config)
        >>> trainer = DrQTrainer(config, agent, env=env)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: RLConfig,
        policy: DrQAgent,
        replay_buffer: Optional[RolloutReplayBuffer] = None,
        env=None,
        eval_env=None,
        image_pad: int = 4,
        image_size: int = 84,
    ):
        """
        Initialize DrQ trainer.
        
        Args:
            config: RL configuration
            policy: DrQ agent (implements RLPolicy interface)
            replay_buffer: Optional replay buffer (will create RolloutReplayBuffer if None)
            env: Training environment
            eval_env: Evaluation environment (defaults to env)
            image_pad: Padding for random shift augmentation
            image_size: Image observation size
        """
        # Store DrQ-specific params before super().__init__
        self.image_pad = image_pad
        self.image_size = image_size
        
        # Create replay buffer if not provided
        if replay_buffer is None and env is not None:
            replay_buffer = RolloutReplayBuffer(
                capacity=config.buffer_capacity,
                num_envs=config.num_envs,
                chunk_size=config.chunk_size,
                ctrl_space=config.ctrl_space,
                ctrl_type=config.ctrl_type,
                device=config.device,
                storage_device=config.buffer_device,
            )
        
        # Initialize parent
        super().__init__(
            config=config,
            policy=policy,
            replay_buffer=replay_buffer,
            env=env,
            eval_env=eval_env,
        )
        
        # Create augmentation module
        self.augmentation = create_augmentation(image_size, image_pad)
        self.augmentation.to(config.device)
        
        # Store DrQ agent reference
        self.drq_agent = policy
        
        # Video recording settings
        self.save_video = True
        self.video_fps = 30

    def _train_online(self):
        """
        Override online training to use DrQ-specific update.
        
        Key differences from base RLTrainer:
        1. Apply augmentation before update
        2. Handle image observations from DMC environments
        3. Support done_no_max for proper bootstrapping
        """
        if self.env is None:
            raise RuntimeError("Environment not provided for online training")
        
        num_envs = self.config.num_envs
        logger.info(f"Starting DrQ online training with {num_envs} parallel environment(s)")
        logger.info(f"Image size: {self.image_size}, Augmentation pad: {self.image_pad}")
        
        # Ensure we have RolloutReplayBuffer
        if not isinstance(self.replay_buffer, RolloutReplayBuffer):
            logger.info("Switching to RolloutReplayBuffer for DrQ")
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
        env_done = np.zeros(num_envs, dtype=bool)
        
        from tqdm import tqdm
        pbar = tqdm(total=self.config.total_steps, desc="DrQ Training")
        pbar.update(self.global_step)
        
        # Time tracking
        time_stats = {
            'action_selection': [],
            'env_step': [],
            'buffer_add': [],
            'sample': [],
            'prepare_batch': [],
            'update': [],
            'total': [],
        }
        
        # Detailed timing for prepare_batch and update
        detailed_timing = {
            'prepare_extract': [],
            'prepare_convert': [],
            'prepare_aug': [],
            'prepare_other': [],
            'update_critic': [],
            'update_actor': [],
            'update_target': [],
        }
        
        while self.global_step < self.config.total_steps:
            step_start = time.perf_counter()
            
            # Select action
            action_start = time.perf_counter()
            if self.global_step < self.config.warmup_steps:
                # Random action during warmup
                action_batch = self._sample_random_action_batch(num_envs)
            else:
                # Use DrQ agent's action selection
                self.drq_agent.eval()
                # Extract raw image for DrQ agent
                obs_image = self._extract_image_for_drq(obs)
                if num_envs == 1:
                    action_batch = np.array([self.drq_agent.act(obs_image, sample=True)])
                else:
                    # For vectorized envs, process each observation
                    actions = []
                    for i in range(num_envs):
                        single_obs = self._extract_single_obs_from_batch(obs_image, i)
                        action = self.drq_agent.act(single_obs, sample=True)
                        actions.append(action)
                    action_batch = np.array(actions)
                self.drq_agent.train()
            time_stats['action_selection'].append(time.perf_counter() - action_start)
            
            # Environment step
            env_start = time.perf_counter()
            result = self.env.step(action_batch[0] if num_envs == 1 else action_batch)
            if len(result) == 4:
                next_obs, reward, done, info = result
            else:
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            time_stats['env_step'].append(time.perf_counter() - env_start)
            
            next_obs_dict = self._obs_to_dict(next_obs)
            reward = np.atleast_1d(reward)
            done = np.atleast_1d(done)
            
            # Handle done_no_max for proper bootstrapping
            # Episode ends due to max steps shouldn't be treated as terminal
            max_steps = getattr(self.env, '_max_episode_steps', self.config.max_episode_steps)
            done_no_max = np.array([
                0 if episode_lengths[i] + 1 == max_steps else done[i]
                for i in range(num_envs)
            ])
            
            # Add to buffer
            buffer_start = time.perf_counter()
            self.replay_buffer.add_from_parallel_envs(
                obs_batch=obs_dict,
                action_batch=action_batch,
                reward_batch=reward,
                next_obs_batch=next_obs_dict,
                done_batch=done,
                truncated_batch=done_no_max == 0,  # Mark as truncated if done due to max steps
                skip_mask=env_done,
            )
            time_stats['buffer_add'].append(time.perf_counter() - buffer_start)
            
            # Update per-env statistics
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
            
            # Mark environments as done
            env_done = env_done | done
            
            # Reset statistics for done environments
            episode_returns[done] = 0
            episode_lengths[done] = 0
            env_done[done] = False
            
            # Move to next state
            obs = next_obs
            obs_dict = next_obs_dict
            
            # Policy update with DrQ augmentation
            if self.global_step >= self.config.warmup_steps and self.replay_buffer.size >= self.config.batch_size:
                for _ in range(self.config.env_steps_per_train):
                    # Sample batch from replay buffer (fast path for image-only)
                    sample_start = time.perf_counter()
                    if hasattr(self.replay_buffer, "sample_fast"):
                        batch = self.replay_buffer.sample_fast(
                            self.config.batch_size,
                            keys=["image"],
                            device=self.config.device,
                        )
                    else:
                        batch = self.replay_buffer.sample(self.config.batch_size)
                    time_stats['sample'].append(time.perf_counter() - sample_start)
                    
                    # Convert to DrQ format and apply augmentation
                    prepare_start = time.perf_counter()
                    drq_batch = self._prepare_drq_batch(batch, detailed_timing=detailed_timing)
                    time_stats['prepare_batch'].append(time.perf_counter() - prepare_start)
                    
                    # Update DrQ agent
                    update_start = time.perf_counter()
                    metrics = self.drq_agent.update(drq_batch, detailed_timing=detailed_timing)
                    time_stats['update'].append(time.perf_counter() - update_start)
                    
                    for k, v in metrics.items():
                        self.train_metrics.setdefault(k, []).append(v)
            
            step_total = time.perf_counter() - step_start
            time_stats['total'].append(step_total)
            
            self.global_step += num_envs
            pbar.update(num_envs)
            
            # Logging with timing
            if self.global_step % self.config.log_freq == 0:
                self._log_metrics()
                self._log_timing_stats(time_stats, detailed_timing)
                # Reset time stats for next logging period
                for key in time_stats:
                    time_stats[key] = []
                for key in detailed_timing:
                    detailed_timing[key] = []
            
            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                self._evaluate_drq()
            
            # Checkpointing
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                )
        
        pbar.close()

    def _prepare_drq_batch(self, batch, detailed_timing: Optional[Dict[str, list]] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for DrQ update with augmentation.
        
        Args:
            batch: BatchTransition from replay buffer
            detailed_timing: Optional dict to collect detailed timing stats
            
        Returns:
            Dictionary with augmented observations ready for DrQ
        """
        device = torch.device(self.config.device)
        
        # Extract image observations
        extract_start = time.perf_counter()
        if hasattr(batch, 'obs'):
            obs = batch.obs
            next_obs = batch.next_obs
        else:
            obs = batch['obs']
            next_obs = batch['next_obs']
        
        # Handle dict observations (extract image)
        if isinstance(obs, dict):
            obs_image = obs.get('image', obs.get('obs'))
            next_obs_image = next_obs.get('image', next_obs.get('obs'))
        else:
            obs_image = obs
            next_obs_image = next_obs
        if detailed_timing:
            detailed_timing.setdefault('prepare_extract', []).append(time.perf_counter() - extract_start)
        
        # Convert to tensor if needed
        convert_start = time.perf_counter()
        if isinstance(obs_image, np.ndarray):
            obs_image = torch.from_numpy(obs_image)
        if isinstance(next_obs_image, np.ndarray):
            next_obs_image = torch.from_numpy(next_obs_image)
        
        # Move to device and ensure float
        obs_image = obs_image.to(device).float()
        next_obs_image = next_obs_image.to(device).float()
        
        # Handle camera dimension: (B, N_cameras, C, H, W) -> (B, C, H, W)
        if obs_image.dim() == 5:
            obs_image = obs_image[:, 0]  # Select first camera
            next_obs_image = next_obs_image[:, 0]
        if detailed_timing:
            detailed_timing.setdefault('prepare_convert', []).append(time.perf_counter() - convert_start)
        
        # Apply DrQ augmentation (4 independent augmentations)
        # Note: We don't need .clone() because augmentation creates new tensors internally
        # The augmentation module uses padding + crop which creates new memory
        aug_start = time.perf_counter()
        self.augmentation.train()
        
        # Augment obs and next_obs (each gets 2 augmented versions for DrQ)
        # Using the same tensor without clone since aug creates new tensors
        obs_aug = self.augmentation(obs_image)
        obs_aug2 = self.augmentation(obs_image)
        next_obs_aug = self.augmentation(next_obs_image)
        next_obs_aug2 = self.augmentation(next_obs_image)
        if detailed_timing:
            detailed_timing.setdefault('prepare_aug', []).append(time.perf_counter() - aug_start)
        
        # Get action, reward, not_done
        other_start = time.perf_counter()
        action = batch.action if hasattr(batch, 'action') else batch['action']
        reward = batch.reward if hasattr(batch, 'reward') else batch['reward']
        done = batch.done if hasattr(batch, 'done') else batch['done']
        
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)
        if isinstance(done, np.ndarray):
            done = torch.from_numpy(done)
        
        action = action.to(device).float()
        reward = reward.to(device).float()
        not_done = (~done.bool()).to(device).float()
        
        # Ensure correct shapes
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if not_done.dim() == 1:
            not_done = not_done.unsqueeze(-1)
        if detailed_timing:
            detailed_timing.setdefault('prepare_other', []).append(time.perf_counter() - other_start)
        
        return {
            'obs': obs_aug,
            'obs_aug': obs_aug2,
            'action': action,
            'reward': reward,
            'next_obs': next_obs_aug,
            'next_obs_aug': next_obs_aug2,
            'not_done': not_done,
        }

    def _evaluate_drq(self):
        """
        Evaluate DrQ agent.
        
        Uses deterministic action selection and optionally records video.
        """
        if self.eval_env is None:
            logger.warning("No evaluation environment, skipping evaluation")
            return
        
        self.drq_agent.eval()
        
        eval_returns = []
        eval_lengths = []
        
        for ep in range(self.config.num_eval_episodes):
            obs = self._reset_env(self.eval_env)
            done = False
            episode_return = 0
            episode_length = 0
            
            # Video recording for first episode
            frames = []
            record_video = self.save_video and ep == 0
            
            while not done and episode_length < self.config.max_episode_steps:
                # Deterministic action - extract image for DrQ
                obs_image = self._extract_image_for_drq(obs)
                action = self.drq_agent.act(obs_image, sample=False)
                
                result = self.eval_env.step(action)
                if len(result) == 4:
                    obs, reward, done, info = result
                else:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                if record_video and hasattr(self.eval_env, 'render'):
                    try:
                        frame = self.eval_env.render(mode='rgb_array')
                        if frame is not None:
                            frames.append(frame)
                    except:
                        pass
            
            eval_returns.append(episode_return)
            eval_lengths.append(episode_length)
            
            # Save video
            if record_video and frames:
                self._save_video(frames, f"eval_{self.global_step}.mp4")
        
        self.drq_agent.train()
        
        # Log metrics
        mean_return = np.mean(eval_returns)
        self.eval_metrics['eval_return'] = mean_return
        self.eval_metrics['eval_length'] = np.mean(eval_lengths)
        
        logger.info(f"Evaluation at step {self.global_step}: "
                   f"mean_return={mean_return:.2f}, "
                   f"mean_length={np.mean(eval_lengths):.1f}")
        
        # Save best model
        if mean_return > self.best_eval_return:
            self.best_eval_return = mean_return
            self.save_checkpoint(os.path.join(self.config.output_dir, "best"))
            logger.info(f"New best model saved with return {mean_return:.2f}")

    def _save_video(self, frames: list, filename: str):
        """Save frames as video."""
        try:
            import imageio
            video_path = os.path.join(self.config.output_dir, 'videos')
            os.makedirs(video_path, exist_ok=True)
            
            with imageio.get_writer(
                os.path.join(video_path, filename),
                fps=self.video_fps
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
                    
            logger.info(f"Saved video to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")

    def _extract_single_obs_from_batch(self, obs, idx: int):
        """Extract single observation from batched observations."""
        if isinstance(obs, dict):
            return {k: v[idx] if hasattr(v, '__getitem__') else v for k, v in obs.items()}
        elif isinstance(obs, (list, tuple)):
            return obs[idx]
        elif isinstance(obs, np.ndarray) and obs.ndim > 3:
            return obs[idx]
        else:
            return obs

    def _extract_image_for_drq(self, obs) -> np.ndarray:
        """
        Extract raw image from observation for DrQ agent.
        
        Handles MetaObs, dict, and raw numpy array observations.
        
        Args:
            obs: Observation (MetaObs, dict, or numpy array)
            
        Returns:
            Raw image as numpy array (C*k, H, W) for DrQ
        """
        # Handle MetaObs
        if hasattr(obs, 'image'):
            image = obs.image
        elif isinstance(obs, dict):
            image = obs.get('image', obs.get('obs', obs.get('pixels')))
        else:
            image = obs
        
        # If image is None, return obs as is
        if image is None:
            return obs
        
        # Convert to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Handle camera dimension: (N_cameras, C*k, H, W) -> (C*k, H, W)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]  # Select first camera
        
        return image

    def _obs_to_dict(self, obs, add_batch_dim: bool = True) -> Dict[str, np.ndarray]:
        """
        Convert observation to dict format for buffer storage.
        
        Handles image observations from DMC environments.
        Adds batch dimension for add_from_parallel_envs compatibility.
        
        Args:
            obs: Observation (numpy array, dict, or dataclass)
            add_batch_dim: If True, add batch dimension for single env case
            
        Returns:
            Dict with observation data, potentially with batch dimension added
        """
        if isinstance(obs, dict):
            result = obs
        elif isinstance(obs, np.ndarray):
            result = {'image': obs}
        elif hasattr(obs, '__dataclass_fields__'):
            from dataclasses import asdict
            result = asdict(obs)
        else:
            result = {'obs': obs}
        
        # Add batch dimension for single env case
        if add_batch_dim:
            result = {k: v[np.newaxis, ...] if isinstance(v, np.ndarray) else v 
                      for k, v in result.items()}
        
        return result

    def _log_timing_stats(self, time_stats: Dict[str, list], detailed_timing: Optional[Dict[str, list]] = None):
        """
        Log timing statistics for performance analysis.
        
        Args:
            time_stats: Dictionary with timing lists for each operation
            detailed_timing: Dictionary with detailed timing for prepare_batch and update
        """
        if not any(time_stats.values()):
            return
        
        log_str = "Timing (ms):"
        
        for key, times in time_stats.items():
            if times:
                mean_time = np.mean(times) * 1000  # Convert to milliseconds
                max_time = np.max(times) * 1000
                min_time = np.min(times) * 1000
                log_str += f" {key}={mean_time:.2f}({min_time:.2f}-{max_time:.2f})"
        
        logger.info(log_str)
        
        # Also log percentage breakdown
        total_times = time_stats.get('total', [])
        if total_times:
            mean_total = np.mean(total_times)
            if mean_total > 0:
                breakdown = []
                for key in ['action_selection', 'env_step', 'buffer_add', 'sample', 'prepare_batch', 'update']:
                    if time_stats.get(key):
                        mean_time = np.mean(time_stats[key])
                        pct = (mean_time / mean_total) * 100
                        breakdown.append(f"{key}={pct:.1f}%")
                
                if breakdown:
                    logger.info(f"Time breakdown: {', '.join(breakdown)}")
        
        # Log detailed timing for prepare_batch
        if detailed_timing:
            prepare_times = time_stats.get('prepare_batch', [])
            if prepare_times and any(detailed_timing.get('prepare_extract', [])):
                mean_prepare = np.mean(prepare_times)
                if mean_prepare > 0:
                    prepare_breakdown = []
                    for key in ['prepare_extract', 'prepare_convert', 'prepare_aug', 'prepare_other']:
                        if detailed_timing.get(key):
                            mean_time = np.mean(detailed_timing[key])
                            pct = (mean_time / mean_prepare) * 100
                            ms = mean_time * 1000
                            prepare_breakdown.append(f"{key}={pct:.1f}%({ms:.2f}ms)")
                    
                    if prepare_breakdown:
                        logger.info(f"  prepare_batch details: {', '.join(prepare_breakdown)}")
            
            # Log detailed timing for update
            update_times = time_stats.get('update', [])
            if update_times and any(detailed_timing.get('update_critic', [])):
                mean_update = np.mean(update_times)
                if mean_update > 0:
                    update_breakdown = []
                    for key in ['update_critic', 'update_actor', 'update_target']:
                        if detailed_timing.get(key):
                            mean_time = np.mean(detailed_timing[key])
                            pct = (mean_time / mean_update) * 100
                            ms = mean_time * 1000
                            update_breakdown.append(f"{key}={pct:.1f}%({ms:.2f}ms)")
                    
                    if update_breakdown:
                        logger.info(f"  update details: {', '.join(update_breakdown)}")

    def save_checkpoint(self, save_dir: str):
        """Save DrQ checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save DrQ agent
        self.drq_agent.save(os.path.join(save_dir, 'agent.pt'))
        
        # Save trainer state
        state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'best_eval_return': self.best_eval_return,
            'train_metrics': {k: v[-1000:] for k, v in self.train_metrics.items()},
            'eval_metrics': self.eval_metrics,
            'config': self.config.to_dict(),
        }
        torch.save(state, os.path.join(save_dir, 'trainer_state.pt'))
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_dir}")

    def load_checkpoint(self, load_dir: str):
        """Load DrQ checkpoint."""
        # Load agent
        agent_path = os.path.join(load_dir, 'agent.pt')
        if os.path.exists(agent_path):
            self.drq_agent.load(agent_path)
        
        # Load trainer state
        state_path = os.path.join(load_dir, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.config.device)
            self.global_step = state.get('global_step', 0)
            self.episode_count = state.get('episode_count', 0)
            self.best_eval_return = state.get('best_eval_return', -float('inf'))
            self.train_metrics = state.get('train_metrics', {})
            self.eval_metrics = state.get('eval_metrics', {})
        
        logger.info(f"Checkpoint loaded from {load_dir}, step={self.global_step}")


# ============================================================================
# ILStudio Integration - Trainer class for policy loading system
# ============================================================================

class Trainer(DrQTrainer):
    """
    ILStudio-compatible trainer interface.
    
    This class provides compatibility with ILStudio's training pipeline,
    matching the interface expected by get_policy_trainer_class().
    """
    
    def __init__(self, model_components: Dict[str, Any], args):
        """
        Initialize from ILStudio args.
        
        Args:
            model_components: Dictionary with 'agent' (DrQAgent)
            args: Training arguments
        """
        agent = model_components.get('agent')
        if agent is None:
            raise ValueError("model_components must contain 'agent'")
        
        # Get environment from args
        env = getattr(args, 'env', None)
        eval_env = getattr(args, 'eval_env', env)
        
        if env is None:
            raise ValueError("args must have 'env' attribute")
        
        # Create RLConfig from args
        config = RLConfig(
            mode=RLMode.ONLINE,
            buffer_capacity=getattr(args, 'replay_buffer_capacity', 100000),
            buffer_device=getattr(args, 'buffer_device', 'cpu'),
            batch_size=getattr(args, 'batch_size', 512),
            learning_rate=getattr(args, 'lr', 1e-3),
            discount=getattr(args, 'discount', 0.99),
            tau=getattr(args, 'critic_tau', 0.01),
            total_steps=getattr(args, 'num_train_steps', 1_000_000),
            eval_freq=getattr(args, 'eval_freq', 10000),
            save_freq=getattr(args, 'save_freq', 50000),
            log_freq=getattr(args, 'log_freq', 1000),
            warmup_steps=getattr(args, 'num_seed_steps', 1000),
            env_steps_per_train=getattr(args, 'num_train_iters', 1),
            num_envs=1,  # DrQ typically uses single env
            max_episode_steps=getattr(args, 'max_episode_steps', 1000),
            num_eval_episodes=getattr(args, 'num_eval_episodes', 10),
            output_dir=getattr(args, 'output_dir', 'ckpt/drq_output'),
            seed=getattr(args, 'seed', 0),
            device=getattr(args, 'device', 'cuda'),
            ctrl_space=getattr(args, 'ctrl_space', 'joint'),
            ctrl_type=getattr(args, 'ctrl_type', 'abs'),
            chunk_size=1,
        )
        
        # Get DrQ-specific params
        image_pad = getattr(args, 'image_pad', 4)
        image_size = getattr(args, 'image_size', 84)
        
        # Initialize parent DrQTrainer
        super().__init__(
            config=config,
            policy=agent,
            env=env,
            eval_env=eval_env,
            image_pad=image_pad,
            image_size=image_size,
        )
        
        # Video settings from args
        self.save_video = getattr(args, 'save_video', True)
        self.video_fps = getattr(args, 'video_fps', 30)
    
    def is_world_process_zero(self) -> bool:
        """Check if this is the main process."""
        return True
    
    def save_model(self, output_dir: str):
        """Save model (ILStudio interface)."""
        self.save_checkpoint(output_dir)
    
    def save_state(self):
        """Save training state (ILStudio interface)."""
        self.save_checkpoint(os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}"))
