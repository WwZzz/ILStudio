from .modeling_pi0 import PI0FlowMatching, PI0FlowMatchingConfig
from .data_utils import PI0Processor, data_collator, get_data_processor, get_data_collator
from .trainer import PI0Trainer, create_pi0_trainer, compute_metrics
from transformers import AutoTokenizer
import torch


def load_model(args):
    """Load PI0 model components."""
    # Create configuration
    config = PI0FlowMatchingConfig(
        state_dim=getattr(args, 'state_dim', 14),
        action_dim=getattr(args, 'action_dim', 14),
        camera_names=getattr(args, 'camera_names', ['primary']),
        chunk_size=getattr(args, 'chunk_size', 50),
        n_action_steps=getattr(args, 'n_action_steps', 50),
        freeze_vision_encoder=getattr(args, 'freeze_vision_encoder', True),
        train_expert_only=getattr(args, 'train_expert_only', False),
        train_state_proj=getattr(args, 'train_state_proj', True),
    )
    
    # Create model
    if getattr(args, 'is_pretrained', False):
        model = PI0FlowMatching.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        model = PI0FlowMatching(config)
    
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
    """Get data processor for PI0."""
    from .data_utils import get_data_processor as _get_data_processor
    return _get_data_processor(args, model_components)


def get_data_collator(args, model_components):
    """Get data collator for PI0."""
    from .data_utils import get_data_collator as _get_data_collator
    return _get_data_collator(args, model_components)


def get_trainer_class(args):
    """Get trainer class for PI0."""
    return PI0Trainer


__all__ = [
    'PI0FlowMatching',
    'PI0FlowMatchingConfig',
    'PI0Processor',
    'PI0Trainer',
    'create_pi0_trainer',
    'compute_metrics',
    'load_model',
    'get_data_processor',
    'get_data_collator',
    'get_trainer_class',
    'data_collator'
]

