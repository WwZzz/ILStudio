#!/usr/bin/env python3
"""
IQL Training Script for ILStudio

This script trains IQL on offline datasets using ILStudio's ReplayBuffer.

Usage:
    # Train on D4RL dataset
    python train_iql.py -e halfcheetah-medium-v2 -o ckpt/iql_halfcheetah
    
    # With custom hyperparameters
    python train_iql.py -e walker2d-medium-expert-v2 --tau 0.9 --beta 5.0
    
    # Deterministic policy
    python train_iql.py -e hopper-medium-v2 --deterministic
"""

import os
import argparse
from loguru import logger
import torch
import numpy as np

# Set number of threads for reproducibility
torch.set_num_threads(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train IQL on offline datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument('-e', '--env', type=str, default='halfcheetah-medium-v2',
                       help='D4RL environment name')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/iql_output',
                       help='Output directory for checkpoints')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=1000000,
                       help='Total number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # IQL Hyperparameters
    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.7,
                       help='Expectile for asymmetric loss')
    parser.add_argument('--beta', type=float, default=3.0,
                       help='Temperature for advantage weighting')
    parser.add_argument('--alpha', type=float, default=0.005,
                       help='EMA coefficient for target network')
    
    # Model Architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--n_hidden', type=int, default=2,
                       help='Number of hidden layers')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic policy')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--max_episode_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=1000,
                       help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Checkpoint save frequency')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser.parse_args()


def set_seed(seed: int, env=None):
    """Set random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if env is not None and hasattr(env, 'seed'):
        env.seed(seed)


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    # Load D4RL dataset into ReplayBuffer
    from policy.rl.offline import load_d4rl_to_replay_buffer
    
    logger.info(f"Loading D4RL dataset: {args.env}")
    eval_env, replay_buffer = load_d4rl_to_replay_buffer(
        env_name=args.env,
        device=args.device,
        storage_device='cpu',
    )
    
    # Get dimensions from buffer
    obs_dim = replay_buffer.obs_storage['state'].shape[1]
    action_dim = replay_buffer.actions.shape[-1]
    dataset_size = replay_buffer.size
    
    set_seed(args.seed, eval_env)
    
    # Log config
    logger.info("=" * 60)
    logger.info("IQL Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    logger.info(f"Dataset size: {dataset_size:,}")
    logger.info(f"Steps: {args.num_steps:,}, Batch: {args.batch_size}")
    logger.info(f"IQL: tau={args.tau}, beta={args.beta}, alpha={args.alpha}")
    logger.info("=" * 60)
    
    # Create agent
    from policy.rl.offline.iql import IQLConfig, IQLAgent, IQLTrainer
    from policy.rl.base import RLConfig, RLMode
    
    iql_config = IQLConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        discount=args.discount,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=args.num_steps,
        deterministic_policy=args.deterministic,
        device=args.device,
    )
    agent = IQLAgent(iql_config)
    logger.info(f"Agent params: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Create trainer
    rl_config = RLConfig(
        mode=RLMode.OFFLINE,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        discount=args.discount,
        total_steps=args.num_steps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        num_eval_episodes=args.num_eval_episodes,
        max_episode_steps=args.max_episode_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    
    trainer = IQLTrainer(
        config=rl_config,
        policy=agent,
        replay_buffer=replay_buffer,
        eval_env=eval_env,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()
    
    if eval_env is not None:
        eval_env.close()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
