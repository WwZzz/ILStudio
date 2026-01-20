#!/usr/bin/env python3
"""
DrQ Training Script for ILStudio

This script provides an easy-to-use interface for training DrQ on DMC environments.
Uses RolloutReplayBuffer and inherits from RLTrainer.

Usage:
    # Basic training (single environment)
    python train_drq.py -e cartpole_swingup -o ckpt/drq_cartpole
    
    # Parallel training (4 environments)
    python train_drq.py -e cartpole_swingup -n 4 -o ckpt/drq_cartpole_parallel
    
    # With custom hyperparameters
    python train_drq.py -e cheetah_run --batch_size 512 --action_repeat 8
    
    # Resume training
    python train_drq.py -e walker_walk --resume ckpt/drq_walker/checkpoint-100000
"""

import os
import argparse
from types import SimpleNamespace
from loguru import logger
import torch

# Enable CUDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train DrQ on DMC environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # =========================================================================
    # Environment
    # =========================================================================
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('-e', '--env', type=str, default='cartpole_swingup',
                          help='DMC environment name (e.g., cheetah_run, walker_walk)')
    env_group.add_argument('-n', '--num_envs', type=int, default=1,
                          help='Number of parallel environments')
    env_group.add_argument('--seed', type=int, default=0,
                          help='Random seed')
    env_group.add_argument('--image_size', type=int, default=84,
                          help='Observation image size')
    env_group.add_argument('--action_repeat', type=int, default=4,
                          help='Number of times to repeat each action')
    env_group.add_argument('--frame_stack', type=int, default=3,
                          help='Number of frames to stack')
    
    # =========================================================================
    # Output
    # =========================================================================
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output_dir', type=str, default='ckpt/drq_output',
                             help='Output directory for checkpoints')
    output_group.add_argument('--save_video', action='store_true', default=True,
                             help='Save evaluation videos')
    
    # =========================================================================
    # Training
    # =========================================================================
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--num_train_steps', type=int, default=1000000,
                            help='Total number of training steps')
    train_group.add_argument('--num_seed_steps', type=int, default=1000,
                            help='Number of random exploration steps')
    train_group.add_argument('--num_train_iters', type=int, default=1,
                            help='Number of training iterations per environment step')
    train_group.add_argument('--replay_buffer_capacity', type=int, default=100000,
                            help='Replay buffer capacity')
    train_group.add_argument('--batch_size', type=int, default=128,
                            help='Batch size (128 default, 512 for paper reproduction)')
    train_group.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate')
    train_group.add_argument('--discount', type=float, default=0.99,
                            help='Discount factor')
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--eval_freq', type=int, default=10000,
                           help='Evaluation frequency (steps)')
    eval_group.add_argument('--num_eval_episodes', type=int, default=10,
                           help='Number of evaluation episodes')
    
    # =========================================================================
    # Logging
    # =========================================================================
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log_freq', type=int, default=1000,
                          help='Logging frequency (steps)')
    log_group.add_argument('--save_freq', type=int, default=50000,
                          help='Checkpoint save frequency (steps)')
    
    # =========================================================================
    # Model Architecture
    # =========================================================================
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--feature_dim', type=int, default=50,
                            help='Encoder feature dimension')
    model_group.add_argument('--hidden_dim', type=int, default=1024,
                            help='Hidden layer dimension')
    model_group.add_argument('--hidden_depth', type=int, default=2,
                            help='Number of hidden layers')
    
    # =========================================================================
    # SAC Hyperparameters
    # =========================================================================
    sac_group = parser.add_argument_group('SAC Hyperparameters')
    sac_group.add_argument('--init_temperature', type=float, default=0.1,
                          help='Initial SAC temperature')
    sac_group.add_argument('--actor_update_freq', type=int, default=2,
                          help='Actor update frequency')
    sac_group.add_argument('--critic_tau', type=float, default=0.01,
                          help='Soft update coefficient for target network')
    sac_group.add_argument('--critic_target_update_freq', type=int, default=2,
                          help='Target network update frequency')
    sac_group.add_argument('--image_pad', type=int, default=4,
                          help='Padding for random shift augmentation')
    
    # =========================================================================
    # Device & Resume
    # =========================================================================
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--device', type=str, default='cuda',
                           help='Device (cuda or cpu)')
    misc_group.add_argument('--resume', type=str, default=None,
                           help='Resume from checkpoint directory')
    
    return parser.parse_args()


def create_environments(args):
    """
    Create training and evaluation environments.
    
    Supports parallel environments via VectorDMCEnv when num_envs > 1.
    
    Args:
        args: Parsed arguments
        
    Returns:
        tuple: (env, eval_env, obs_shape, action_shape, action_range, max_episode_steps)
    """
    from benchmark.dmc import make_dmc_env, make_vector_dmc_env
    
    if args.num_envs > 1:
        # Parallel environments
        logger.info(f"Creating {args.num_envs} parallel training environments...")
        env = make_vector_dmc_env(
            env_name=args.env,
            num_envs=args.num_envs,
            seed=args.seed,
            image_size=args.image_size,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
        )
    else:
        # Single environment
        logger.info("Creating training environment...")
        env = make_dmc_env(
            env_name=args.env,
            seed=args.seed,
            image_size=args.image_size,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
        )
    
    # Evaluation environment (always single)
    logger.info("Creating evaluation environment...")
    eval_env = make_dmc_env(
        env_name=args.env,
        seed=args.seed + 100,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
    )
    
    # Extract environment info
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_range = (
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    )
    max_episode_steps = getattr(env, '_max_episode_steps', 1000)
    
    return env, eval_env, obs_shape, action_shape, action_range, max_episode_steps


def create_agent(args, obs_shape, action_dim, action_range):
    """
    Create DrQ agent.
    
    Args:
        args: Parsed arguments
        obs_shape: Observation shape
        action_dim: Action dimension
        action_range: (action_low, action_high)
        
    Returns:
        DrQAgent instance
    """
    from policy.rl.online.drq.modeling import DrQConfig, DrQAgent
    
    config = DrQConfig(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_range=action_range,
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
    
    agent = DrQAgent(config)
    num_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Created DrQ agent with {num_params:,} parameters")
    
    return agent


def create_trainer(args, agent, env, eval_env, max_episode_steps):
    """
    Create DrQ trainer.
    
    Args:
        args: Parsed arguments
        agent: DrQ agent
        env: Training environment
        eval_env: Evaluation environment
        max_episode_steps: Maximum steps per episode
        
    Returns:
        DrQTrainer instance
    """
    from policy.rl.online.drq.trainer import DrQTrainer
    from policy.rl.base import RLConfig, RLMode
    
    config = RLConfig(
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
        num_envs=args.num_envs,
        max_episode_steps=max_episode_steps,
        num_eval_episodes=args.num_eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        ctrl_space='joint',
        ctrl_type='abs',
        chunk_size=1,
    )
    
    trainer = DrQTrainer(
        config=config,
        policy=agent,
        env=env,
        eval_env=eval_env,
        image_pad=args.image_pad,
        image_size=args.image_size,
    )
    trainer.save_video = args.save_video
    
    return trainer


def log_config(args, obs_shape, action_shape, action_range, max_episode_steps):
    """Log training configuration."""
    logger.info("=" * 60)
    logger.info("DrQ Training Configuration")
    logger.info("=" * 60)
    
    # Environment
    logger.info("Environment:")
    logger.info(f"  Task: {args.env}")
    logger.info(f"  Num envs: {args.num_envs}")
    logger.info(f"  Observation shape: {obs_shape}")
    logger.info(f"  Action shape: {action_shape}")
    logger.info(f"  Action range: [{action_range[0]:.1f}, {action_range[1]:.1f}]")
    logger.info(f"  Max episode steps: {max_episode_steps}")
    logger.info(f"  Action repeat: {args.action_repeat}")
    logger.info(f"  Frame stack: {args.frame_stack}")
    
    # Training
    logger.info("Training:")
    logger.info(f"  Total steps: {args.num_train_steps:,}")
    logger.info(f"  Warmup steps: {args.num_seed_steps:,}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Discount: {args.discount}")
    logger.info(f"  Buffer capacity: {args.replay_buffer_capacity:,}")
    
    # SAC
    logger.info("SAC:")
    logger.info(f"  Init temperature: {args.init_temperature}")
    logger.info(f"  Actor update freq: {args.actor_update_freq}")
    logger.info(f"  Critic tau: {args.critic_tau}")
    logger.info(f"  Image pad: {args.image_pad}")
    
    # Output
    logger.info("Output:")
    logger.info(f"  Directory: {args.output_dir}")
    logger.info(f"  Log freq: {args.log_freq}")
    logger.info(f"  Save freq: {args.save_freq}")
    logger.info(f"  Eval freq: {args.eval_freq}")
    
    logger.info("=" * 60)


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environments
    env, eval_env, obs_shape, action_shape, action_range, max_episode_steps = \
        create_environments(args)
    
    # Log configuration
    log_config(args, obs_shape, action_shape, action_range, max_episode_steps)
    
    # Create agent
    agent = create_agent(
        args=args,
        obs_shape=obs_shape,
        action_dim=action_shape[0],
        action_range=action_range,
    )
    
    # Create trainer
    trainer = create_trainer(
        args=args,
        agent=agent,
        env=env,
        eval_env=eval_env,
        max_episode_steps=max_episode_steps,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Cleanup
    if hasattr(env, 'close'):
        env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
