#!/usr/bin/env python3
"""
DrQ Training Script for ILStudio

This script provides an easy-to-use interface for training DrQ on DMC environments.
Uses RolloutReplayBuffer and inherits from RLTrainer.

Usage:
    # Basic training (should achieve SOTA in ~3 hours on cartpole_swingup)
    python train_drq.py --env cartpole_swingup -o ckpt/drq_cartpole
    
    # With custom hyperparameters (paper reproduction settings)
    python train_drq.py --env cheetah_run \
        --num_train_steps 1000000 \
        --batch_size 512 \
        --action_repeat 8
    
    # Resume training
    python train_drq.py --env walker_walk --resume ckpt/drq_walker/checkpoint-100000
"""

import os
import argparse
from types import SimpleNamespace
from loguru import logger
import torch

# Enable CUDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True

from policy.rl.online.drq.modeling import DrQConfig, DrQAgent
from policy.rl.online.drq.trainer import DrQTrainer
from policy.rl.base import RLConfig, RLMode
from benchmark.dmc import create_env, make_dmc_env


def parse_args():
    parser = argparse.ArgumentParser(description='Train DrQ on DMC environments')
    
    # Environment
    parser.add_argument('--env', '-e', type=str, default='cartpole_swingup',
                       help='DMC environment name (e.g., cheetah_run, walker_walk)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--image_size', type=int, default=84, help='Observation image size')
    parser.add_argument('--action_repeat', type=int, default=4, 
                       help='Number of times to repeat each action (original DrQ uses 4)')
    parser.add_argument('--frame_stack', type=int, default=3,
                       help='Number of frames to stack')
    parser.add_argument('--use_metaenv', action='store_true', default=False,
                       help='Use MetaEnv wrapper (for ILStudio compatibility)')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/drq_output',
                       help='Output directory for checkpoints')
    
    # Training
    parser.add_argument('--num_train_steps', type=int, default=1000000,
                       help='Total number of training steps')
    parser.add_argument('--num_seed_steps', type=int, default=1000,
                       help='Number of random exploration steps')
    parser.add_argument('--num_train_iters', type=int, default=1,
                       help='Number of training iterations per step')
    parser.add_argument('--replay_buffer_capacity', type=int, default=100000,
                       help='Replay buffer capacity')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save_video', action='store_true', default=True,
                       help='Save evaluation videos')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=1000,
                       help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Checkpoint save frequency')
    
    # Model architecture
    parser.add_argument('--feature_dim', type=int, default=50,
                       help='Encoder feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden layer dimension')
    parser.add_argument('--hidden_depth', type=int, default=2,
                       help='Number of hidden layers')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size (128 is default, use 512 for paper reproduction)')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--init_temperature', type=float, default=0.1,
                       help='Initial SAC temperature')
    parser.add_argument('--actor_update_freq', type=int, default=2,
                       help='Actor update frequency')
    parser.add_argument('--critic_tau', type=float, default=0.01,
                       help='Soft update coefficient for target network')
    parser.add_argument('--critic_target_update_freq', type=int, default=2,
                       help='Target network update frequency')
    parser.add_argument('--image_pad', type=int, default=4,
                       help='Padding for random shift augmentation')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')
    
    return parser.parse_args()


def create_env_config(args, seed_offset: int = 0):
    """Create environment config from args."""
    return SimpleNamespace(
        task=args.env,
        seed=args.seed + seed_offset,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        ctrl_space='joint',
        ctrl_type='abs',
        use_camera=True,
        render_mode='rgb_array',
        normalize_actions=True,
    )


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("DrQ Training (using RolloutReplayBuffer & RLTrainer)")
    logger.info("="*60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Training steps: {args.num_train_steps}")
    logger.info(f"Use MetaEnv: {args.use_metaenv}")
    logger.info("="*60)
    
    # Create environments
    if args.use_metaenv:
        # Use MetaEnv wrapper (ILStudio compatible)
        logger.info("Creating training environment (MetaEnv)...")
        env_config = create_env_config(args, seed_offset=0)
        env = create_env(env_config)
        
        logger.info("Creating evaluation environment (MetaEnv)...")
        eval_env_config = create_env_config(args, seed_offset=100)
        eval_env = create_env(eval_env_config)
        
        # Get observation and action shapes from underlying env
        obs_shape = env.env.observation_space.shape
        action_shape = env.env.action_space.shape
        action_low = env.env.action_space.low.min()
        action_high = env.env.action_space.high.max()
        max_episode_steps = getattr(env, '_max_episode_steps', 1000)
    else:
        # Use raw gym env (for DrQ direct training)
        logger.info("Creating training environment (raw gym)...")
        env = make_dmc_env(
            env_name=args.env,
            seed=args.seed,
            image_size=args.image_size,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
        )
        
        logger.info("Creating evaluation environment (raw gym)...")
        eval_env = make_dmc_env(
            env_name=args.env,
            seed=args.seed + 100,
            image_size=args.image_size,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
        )
        
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        action_low = env.action_space.low.min()
        action_high = env.action_space.high.max()
        max_episode_steps = getattr(env, '_max_episode_steps', 1000)
    
    logger.info(f"Observation space: {obs_shape}")
    logger.info(f"Action space: {action_shape}")
    logger.info(f"Action range: [{action_low}, {action_high}]")
    logger.info(f"Max episode steps: {max_episode_steps}")
    
    # Create DrQ config
    drq_config = DrQConfig(
        obs_shape=obs_shape,
        action_dim=action_shape[0],
        action_range=(float(action_low), float(action_high)),
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        discount=args.discount,
        init_temperature=args.init_temperature,
        lr=args.lr,
        actor_update_frequency=args.actor_update_freq,
        critic_tau=args.critic_tau,
        critic_target_update_frequency=args.critic_target_update_freq,
        batch_size=args.batch_size,
        image_pad=args.image_pad,
        device=args.device,
    )
    
    # Create DrQ agent
    agent = DrQAgent(drq_config)
    logger.info(f"Created DrQ agent with {sum(p.numel() for p in agent.parameters())} parameters")
    
    # Create RL config for trainer
    rl_config = RLConfig(
        mode=RLMode.ONLINE,
        buffer_capacity=args.replay_buffer_capacity,
        buffer_device='cpu',
        batch_size=args.batch_size,
        learning_rate=args.lr,
        discount=args.discount,
        tau=args.critic_tau,
        total_steps=args.num_train_steps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        warmup_steps=args.num_seed_steps,
        env_steps_per_train=args.num_train_iters,
        num_envs=1,
        max_episode_steps=max_episode_steps,
        num_eval_episodes=args.num_eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        ctrl_space='joint',
        ctrl_type='abs',
        chunk_size=1,
    )
    
    # Create trainer
    trainer = DrQTrainer(
        config=rl_config,
        policy=agent,
        env=env,
        eval_env=eval_env,
        image_pad=args.image_pad,
        image_size=args.image_size,
    )
    trainer.save_video = args.save_video
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    # Cleanup
    if hasattr(env, 'close'):
        env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
