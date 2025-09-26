# SmolVLA policy module for IL-Studio
from .configuration import SmolVLAConfig
from .modeling import SmolVLAPolicy, VLAFlowMatching
from .data_utils import SmolVLAProcessor, data_collator, get_data_processor, get_data_collator
from .trainer import SmolVLATrainer, compute_metrics
from .smolvlm_with_expert import SmolVLMWithExpertModel

import torch
from transformers import AutoProcessor


def load_model(args):
    """Load SmolVLA model components following IL-Studio policy rules."""
    from .configuration import SmolVLAConfig
    from .modeling import SmolVLAPolicy
    
    # Extract parameters from args with model_args precedence (following IL-Studio conventions)
    def get_param(name, default):
        return getattr(args, name, default)
    
    # Create configuration
    config = SmolVLAConfig(
        # Basic dimensions
        state_dim=get_param('state_dim', 14),
        action_dim=get_param('action_dim', 14),
        camera_names=get_param('camera_names', ['primary']),
        chunk_size=get_param('chunk_size', 50),
        n_action_steps=get_param('n_action_steps', 50),
        
        # Model parameters
        vlm_model_name=get_param('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'),
        load_vlm_weights=get_param('load_vlm_weights', False),
        freeze_vision_encoder=get_param('freeze_vision_encoder', True),
        train_expert_only=get_param('train_expert_only', True),
        train_state_proj=get_param('train_state_proj', True),
        
        # Training parameters
        optimizer_lr=get_param('optimizer_lr', 1e-4),
        optimizer_betas=get_param('optimizer_betas', (0.9, 0.95)),
        optimizer_eps=get_param('optimizer_eps', 1e-8),
        optimizer_weight_decay=get_param('optimizer_weight_decay', 1e-10),
        optimizer_grad_clip_norm=get_param('optimizer_grad_clip_norm', 10),
        
        # Architecture parameters
        num_vlm_layers=get_param('num_vlm_layers', 16),
        num_expert_layers=get_param('num_expert_layers', -1),
        attention_mode=get_param('attention_mode', 'cross_attn'),
        expert_width_multiplier=get_param('expert_width_multiplier', 0.75),
        
        # Processing parameters
        max_state_dim=get_param('max_state_dim', 32),
        max_action_dim=get_param('max_action_dim', 32),
        resize_imgs_with_padding=get_param('resize_imgs_with_padding', (512, 512)),
        tokenizer_max_length=get_param('tokenizer_max_length', 48),
        num_steps=get_param('num_steps', 10),
        use_cache=get_param('use_cache', True),
        
        # Special tokens and padding
        add_image_special_tokens=get_param('add_image_special_tokens', False),
        pad_language_to=get_param('pad_language_to', 'longest'),
        empty_cameras=get_param('empty_cameras', 0),
        
        # Aloha adaptation
        adapt_to_pi_aloha=get_param('adapt_to_pi_aloha', False),
        use_delta_joint_actions_aloha=get_param('use_delta_joint_actions_aloha', False),
    )
    
    # Create model
    model = SmolVLAPolicy(config)
    
    # Load pretrained weights if available
    if hasattr(args, 'model_name_or_path') and args.model_name_or_path and args.model_name_or_path != "scratch":
        try:
            # Try to load from checkpoint
            checkpoint = torch.load(args.model_name_or_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {args.model_name_or_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {args.model_name_or_path}: {e}")
            print("Training from scratch...")
    
    # Get tokenizer
    tokenizer = model.language_tokenizer
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config
    }


def get_data_processor(args, model_components):
    """Get data processor for SmolVLA following IL-Studio policy rules."""
    tokenizer = model_components.get('tokenizer')
    if tokenizer is None:
        raise ValueError("Tokenizer not found in model components")
    
    config = model_components.get('config')
    if config is None:
        raise ValueError("Config not found in model components")
    
    return SmolVLAProcessor(tokenizer=tokenizer, config=config)


def get_data_collator(args, model_components):
    """Get data collator for SmolVLA following IL-Studio policy rules."""
    from .data_utils import data_collator
    return data_collator


# Optional: return SmolVLATrainer if available, otherwise use default
Trainer = SmolVLATrainer


__all__ = [
    'SmolVLAConfig',
    'SmolVLAPolicy', 
    'VLAFlowMatching',
    'SmolVLAProcessor',
    'SmolVLATrainer',
    'compute_metrics',
    'SmolVLMWithExpertModel',
    'load_model',
    'get_data_processor',
    'get_data_collator',
    'data_collator'
]

