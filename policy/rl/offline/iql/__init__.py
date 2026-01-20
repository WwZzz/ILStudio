"""
IQL (Implicit Q-Learning) for ILStudio.

IQL is an offline reinforcement learning algorithm that learns from fixed datasets
without requiring environment interaction during training.

Reference:
    Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning"
    https://arxiv.org/abs/2110.06169

Usage:
    from policy.rl.offline import load_d4rl_to_replay_buffer
    from policy.rl.offline.iql import IQLAgent, IQLConfig, IQLTrainer
    
    # Load D4RL dataset into ReplayBuffer
    env, replay_buffer = load_d4rl_to_replay_buffer('halfcheetah-medium-v2')
    
    # Create agent
    config = IQLConfig(obs_dim=17, action_dim=6)
    agent = IQLAgent(config)
    
    # Create trainer with ReplayBuffer
    trainer = IQLTrainer(rl_config, agent, replay_buffer=replay_buffer)
    trainer.train()
"""

import os
import json
from typing import Dict, Any

from policy.rl.offline.iql.modeling import (
    IQLConfig,
    IQLAgent,
    TwinQ,
    ValueFunction,
    GaussianPolicy,
    DeterministicPolicy,
)
from policy.rl.offline.iql.trainer import IQLTrainer

__all__ = [
    # Config
    'IQLConfig',
    # Agent
    'IQLAgent',
    # Networks
    'TwinQ',
    'ValueFunction',
    'GaussianPolicy',
    'DeterministicPolicy',
    # Trainer
    'IQLTrainer',
]


def load_model(args) -> Dict[str, Any]:
    """
    Load or create IQL model.
    
    This is the main entry point for ILStudio's policy loading system.
    
    Args:
        args: Arguments with model configuration
        
    Returns:
        Dictionary with 'agent' key containing IQLAgent
    """
    if not args.is_training:
        # Load trained model
        checkpoint_path = args.model_name_or_path
        
        # Load config
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Extract IQL config
            iql_config_dict = config_dict.get('iql_config', config_dict)
            
            config = IQLConfig(**iql_config_dict)
        else:
            # Default config
            config = IQLConfig(
                device=getattr(args, 'device', 'cuda'),
            )
        
        # Create agent and load weights
        agent = IQLAgent(config)
        
        # Try different checkpoint locations
        agent_path = os.path.join(checkpoint_path, 'agent.pt')
        if not os.path.exists(agent_path):
            agent_path = os.path.join(checkpoint_path, 'best', 'agent.pt')
        if not os.path.exists(agent_path):
            agent_path = os.path.join(checkpoint_path, 'final', 'agent.pt')
        
        if os.path.exists(agent_path):
            agent.load(agent_path)
            print(f"Loaded IQL agent from {agent_path}")
        else:
            raise FileNotFoundError(f"No agent checkpoint found in {checkpoint_path}")
        
        agent.eval()
        
    else:
        # Create new model for training
        model_args = getattr(args, 'model_args', {})
        
        # Get dimensions from replay buffer if available
        replay_buffer = getattr(args, 'replay_buffer', None)
        if replay_buffer is not None:
            obs_dim = int(replay_buffer.obs_storage['state'].shape[1])
            action_dim = int(replay_buffer.actions.shape[-1])
        else:
            # Defaults
            obs_dim = model_args.get('obs_dim', 17)
            action_dim = model_args.get('action_dim', 6)
        
        config = IQLConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=model_args.get('hidden_dim', 256),
            n_hidden=model_args.get('n_hidden', 2),
            discount=model_args.get('discount', 0.99),
            tau=model_args.get('tau', 0.7),
            beta=model_args.get('beta', 3.0),
            alpha=model_args.get('alpha', 0.005),
            learning_rate=model_args.get('lr', 3e-4),
            batch_size=model_args.get('batch_size', 256),
            max_steps=model_args.get('num_steps', 1000000),
            deterministic_policy=model_args.get('deterministic', False),
            device=getattr(args, 'device', 'cuda'),
        )
        
        agent = IQLAgent(config)
        agent.train()
        
        print(f"Created IQL agent:")
        print(f"  Obs dim: {config.obs_dim}, Action dim: {config.action_dim}")
        print(f"  Hidden dim: {config.hidden_dim}, Num hidden: {config.n_hidden}")
        print(f"  Tau: {config.tau}, Beta: {config.beta}, Alpha: {config.alpha}")
    
    return {'agent': agent}
