"""
DrQ (Data-regularized Q) for ILStudio.

DrQ is an image-based reinforcement learning algorithm that achieves
state-of-the-art performance on DMC (DeepMind Control Suite) by using
random shift augmentation for data regularization.

This implementation:
1. Uses RolloutReplayBuffer from policy.rl.replay_buffer for data storage
2. Inherits from RLTrainer for training loop
3. Applies augmentation during the update step (not during sampling)

Reference: Kostrikov et al., "Image Augmentation Is All You Need: 
Regularizing Deep Reinforcement Learning from Pixels" (2020)

Usage:
    # Training
    python train_drq.py --env cheetah_run -o ckpt/drq_cheetah
    
    # Evaluation
    python eval_sim.py -m ckpt/drq_cheetah -e dmc --env.task cheetah_run
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Any, Optional

from .modeling import DrQAgent, DrQConfig
from .data_utils import DrQProcessor, DrQCollator, create_augmentation
from .trainer import DrQTrainer, Trainer

# Re-export for convenience
from policy.rl.replay_buffer import RolloutReplayBuffer
from policy.rl.base import RLConfig, RLMode, RLTrainer


def load_model(args) -> Dict[str, Any]:
    """
    Load or create DrQ model.
    
    This is the main entry point for ILStudio's policy loading system.
    
    Args:
        args: Arguments with model configuration
        
    Returns:
        Dictionary with 'agent' key containing DrQAgent
    """
    if not args.is_training:
        # Load trained model
        checkpoint_path = args.model_name_or_path
        
        # Load config
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Extract DrQ config
            drq_config_dict = config_dict.get('drq_config', config_dict)
            
            # Handle tuple conversion for obs_shape and action_range
            if 'obs_shape' in drq_config_dict and isinstance(drq_config_dict['obs_shape'], list):
                drq_config_dict['obs_shape'] = tuple(drq_config_dict['obs_shape'])
            if 'action_range' in drq_config_dict and isinstance(drq_config_dict['action_range'], list):
                drq_config_dict['action_range'] = tuple(drq_config_dict['action_range'])
            if 'log_std_bounds' in drq_config_dict and isinstance(drq_config_dict['log_std_bounds'], list):
                drq_config_dict['log_std_bounds'] = tuple(drq_config_dict['log_std_bounds'])
            
            config = DrQConfig(**drq_config_dict)
        else:
            # Default config
            config = DrQConfig(
                device=getattr(args, 'device', 'cuda'),
            )
        
        # Create agent and load weights
        agent = DrQAgent(config)
        
        # Try different checkpoint locations
        agent_path = os.path.join(checkpoint_path, 'agent.pt')
        if not os.path.exists(agent_path):
            agent_path = os.path.join(checkpoint_path, 'best', 'agent.pt')
        if not os.path.exists(agent_path):
            agent_path = os.path.join(checkpoint_path, 'final', 'agent.pt')
        
        if os.path.exists(agent_path):
            agent.load(agent_path)
            print(f"Loaded DrQ agent from {agent_path}")
        else:
            raise FileNotFoundError(f"No agent checkpoint found in {checkpoint_path}")
        
        agent.eval()
        
    else:
        # Create new model for training
        model_args = getattr(args, 'model_args', {})
        
        # Get observation and action dimensions from environment if available
        env = getattr(args, 'env', None)
        if env is not None:
            obs_shape = env.observation_space.shape
            action_dim = env.action_space.shape[0]
            action_range = (
                float(env.action_space.low.min()),
                float(env.action_space.high.max())
            )
        else:
            # Defaults for DMC
            obs_shape = model_args.get('obs_shape', (9, 84, 84))
            action_dim = model_args.get('action_dim', 6)
            action_range = model_args.get('action_range', (-1.0, 1.0))
        
        config = DrQConfig(
            # Observation and action
            obs_shape=obs_shape,
            action_dim=action_dim,
            action_range=action_range,
            
            # Network
            feature_dim=model_args.get('feature_dim', 50),
            hidden_dim=model_args.get('hidden_dim', 1024),
            hidden_depth=model_args.get('hidden_depth', 2),
            
            # Training
            discount=model_args.get('discount', 0.99),
            init_temperature=model_args.get('init_temperature', 0.1),
            lr=model_args.get('lr', 1e-3),
            actor_update_frequency=model_args.get('actor_update_frequency', 2),
            critic_tau=model_args.get('critic_tau', 0.01),
            critic_target_update_frequency=model_args.get('critic_target_update_frequency', 2),
            batch_size=model_args.get('batch_size', 512),
            
            # Augmentation
            image_pad=model_args.get('image_pad', 4),
            
            # Device
            device=getattr(args, 'device', 'cuda'),
        )
        
        agent = DrQAgent(config)
        agent.train()
        
        print(f"Created DrQ agent:")
        print(f"  Observation shape: {config.obs_shape}")
        print(f"  Action dim: {config.action_dim}")
        print(f"  Feature dim: {config.feature_dim}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.lr}")
    
    return {'agent': agent}


def get_data_processor(args, model_components: Dict[str, Any]) -> DrQProcessor:
    """
    Get data processor for DrQ.
    
    Args:
        args: Arguments
        model_components: Model components from load_model
        
    Returns:
        DrQProcessor instance
    """
    model_args = getattr(args, 'model_args', {})
    
    return DrQProcessor(
        image_size=model_args.get('image_size', 84),
        frame_stack=model_args.get('frame_stack', 3),
        image_pad=model_args.get('image_pad', 4),
    )


def get_data_collator(args, model_components: Dict[str, Any]) -> DrQCollator:
    """
    Get data collator for DrQ.
    
    Args:
        args: Arguments
        model_components: Model components from load_model
        
    Returns:
        DrQCollator instance
    """
    return DrQCollator()


# ============================================================================
# Policy Wrapper for Evaluation
# ============================================================================

class DrQPolicyWrapper:
    """
    Wrapper for DrQ agent compatible with ILStudio's evaluation system.
    
    Provides the select_action interface expected by eval_sim.py
    """
    
    def __init__(self, agent: DrQAgent):
        self.agent = agent
        self.policy = agent  # For compatibility
        
    def select_action(self, obs, timestep: int = 0) -> Any:
        """
        Select action given observation.
        
        Args:
            obs: Observation (can be MetaObs, dict, or numpy array)
            timestep: Current timestep (unused for DrQ)
            
        Returns:
            Action array
        """
        # Extract image from various formats
        if hasattr(obs, 'image'):
            # MetaObs
            image = obs.image
        elif isinstance(obs, dict):
            image = obs.get('image', obs.get('obs'))
        else:
            image = obs
        
        # Handle batched observations
        if isinstance(image, (list, tuple)):
            # Process each observation
            actions = []
            for img in image:
                if hasattr(img, 'image'):
                    img = img.image
                elif isinstance(img, dict):
                    img = img.get('image', img.get('obs'))
                
                # Remove camera dimension if present
                if isinstance(img, np.ndarray) and img.ndim == 4:
                    img = img[0]
                elif isinstance(img, torch.Tensor) and img.dim() == 4:
                    img = img[0].numpy()
                
                action = self.agent.act(img, sample=False)
                actions.append(action)
            return np.array(actions)
        else:
            # Single observation
            if isinstance(image, np.ndarray) and image.ndim == 4:
                image = image[0]  # Remove camera dim
            elif isinstance(image, torch.Tensor) and image.dim() == 4:
                image = image[0].numpy()
            
            return self.agent.act(image, sample=False)
    
    def reset(self):
        """Reset policy state (no-op for DrQ)."""
        pass
    
    def eval(self):
        """Set to evaluation mode."""
        self.agent.eval()
    
    def train(self):
        """Set to training mode."""
        self.agent.train()


def create_policy_wrapper(model_components: Dict[str, Any]) -> DrQPolicyWrapper:
    """Create policy wrapper for evaluation."""
    agent = model_components['agent']
    return DrQPolicyWrapper(agent)
