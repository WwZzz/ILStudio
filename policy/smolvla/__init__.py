from .configuration import SmolVLAConfig
from .modeling import SmolVLAPolicy, VLAFlowMatching
from .data_utils import SmolVLAProcessor, data_collator, get_data_processor, get_data_collator
from .trainer import SmolVLATrainer, create_smolvla_trainer, compute_metrics
from .smolvlm_with_expert import SmolVLMWithExpertModel

import torch
from transformers import AutoProcessor


def load_model(args):
    """Load SmolVLA model components."""
    from .configuration import SmolVLAConfig
    from .modeling import SmolVLAPolicy
    
    # Create configuration
    config = SmolVLAConfig(
        state_dim=getattr(args, 'state_dim', 14),
        action_dim=getattr(args, 'action_dim', 14),
        camera_names=getattr(args, 'camera_names', ['primary']),
        chunk_size=getattr(args, 'chunk_size', 50),
        n_action_steps=getattr(args, 'n_action_steps', 50),
        vlm_model_name=getattr(args, 'vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'),
        freeze_vision_encoder=getattr(args, 'freeze_vision_encoder', True),
        train_expert_only=getattr(args, 'train_expert_only', True),
        train_state_proj=getattr(args, 'train_state_proj', True),
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
    """Get data processor for SmolVLA."""
    from .data_utils import get_data_processor as _get_data_processor
    return _get_data_processor(args, model_components)


def get_data_collator(args, model_components):
    """Get data collator for SmolVLA."""
    from .data_utils import get_data_collator as _get_data_collator
    return _get_data_collator(args, model_components)


def get_trainer_class(args):
    """Get trainer class for SmolVLA."""
    return SmolVLATrainer


__all__ = [
    'SmolVLAConfig',
    'SmolVLAPolicy', 
    'VLAFlowMatching',
    'SmolVLAProcessor',
    'SmolVLATrainer',
    'create_smolvla_trainer',
    'compute_metrics',
    'SmolVLMWithExpertModel',
    'load_model',
    'get_data_processor',
    'get_data_collator',
    'get_trainer_class',
    'data_collator'
]
