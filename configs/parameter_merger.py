"""
Parameter merger utility for combining task, model, and training configurations.
This abstracts away the verbose parameter setting in train.py.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
# from configs.model.loader import ModelConfig  # No longer needed
from configs.training.loader import TrainingConfig


def merge_all_parameters(
    task_config: Dict[str, Any],
    policy_config: Dict[str, Any],
    training_config: TrainingConfig,
    args: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Merge task, policy, and training configurations into a single parameter dictionary.
    
    Args:
        task_config: Task-specific configuration dictionary
        policy_config: Policy configuration dictionary (contains model_args)
        training_config: Training configuration object
        args: Optional args object to set attributes on
        
    Returns:
        Dictionary containing all merged parameters
    """
    
    # Task parameters
    task_params = {
        'action_dim': task_config.get('action_dim', 7),
        'state_dim': task_config.get('state_dim', 7),
        'chunk_size': task_config.get('chunk_size', 16),
        'camera_names': task_config.get('camera_names', ['primary']),
        'image_size_primary': task_config.get('image_size_primary', '(256, 256)'),
        'image_size_wrist': task_config.get('image_size_wrist', '(256, 256)'),
        'action_normalize': task_config.get('action_normalize', 'minmax'),
        'state_normalize': task_config.get('state_normalize', 'minmax'),
        'use_reasoning': task_config.get('use_reasoning', False),
        'use_prev_subtask': task_config.get('use_prev_subtask', False)
    }
    
    # Model parameters (from policy config model_args)
    model_args = policy_config.get('model_args', {})
    pretrained_config = policy_config.get('pretrained_config', {})
    
    model_params = {
        # LoRA Configuration (from model_args)
        'lora_enable': model_args.get('lora_enable', False),
        'lora_module': model_args.get('lora_module', 'all'),
        'lora_task_type': model_args.get('lora_task_type', 'CAUSAL_LM'),
        'lora_r': model_args.get('lora_r', 16),
        'lora_alpha': model_args.get('lora_alpha', 32),
        'lora_dropout': model_args.get('lora_dropout', 0.1),
        'lora_weight_path': model_args.get('lora_weight_path', None),
        'lora_bias': model_args.get('lora_bias', 'none'),
        'lora_lr': model_args.get('lora_lr', 0.0002),
        
        # Quantization Configuration (from model_args)
        'use_quantization': model_args.get('use_quantization', False),
        'bits': model_args.get('bits', 4),
        'double_quant': model_args.get('double_quant', True),
        'quant_type': model_args.get('quant_type', 'nf4'),
        
        # Model Architecture (from pretrained_config and model_args)
        'model_name_or_path': pretrained_config.get('model_name_or_path', model_args.get('model_name_or_path', None)),
        'is_pretrained': pretrained_config.get('is_pretrained', model_args.get('is_pretrained', False)),
        'using_ema': model_args.get('using_ema', False),
        'cache_dir': model_args.get('cache_dir', None),
        'flash_attn': model_args.get('flash_attn', False),
        'freeze_vision_tower': model_args.get('freeze_vision_tower', False),
        'freeze_backbone': model_args.get('freeze_backbone', False),
        'tune_mm_mlp_adapter': model_args.get('tune_mm_mlp_adapter', False),
        'llm_loss_weight': model_args.get('llm_loss_weight', 1.0),
        'load_pretrain': model_args.get('load_pretrain', False),
        
        # Data Processing (from model_args, with defaults)
        'lazy_preprocess': model_args.get('lazy_preprocess', False),
        'select_seg_token_mask': model_args.get('select_seg_token_mask', False),
        'is_multimodal': model_args.get('is_multimodal', True),
        'image_aspect_ratio': model_args.get('image_aspect_ratio', 'square'),
        'skip_mirrored_data': model_args.get('skip_mirrored_data', False),
        'history_images_length': model_args.get('history_images_length', 1),
    }
    
    # Training parameters (from training config)
    training_params = {
        # Data loading (training process related)
        'preload_data': training_config.preload_data,
        
        # Logging and saving parameters
        'logging_strategy': training_config.logging_strategy,
        'logging_steps': training_config.logging_steps,
        'report_to': training_config.report_to,
        'save_strategy': training_config.save_strategy,
        'save_steps': training_config.save_steps,
        'save_total_limit': training_config.save_total_limit,
        
        # Data processing parameters (TrainingArguments)
        'dataloader_num_workers': training_config.dataloader_num_workers,
        'dataloader_pin_memory': training_config.dataloader_pin_memory,
        'remove_unused_columns': training_config.remove_unused_columns,
        'do_eval': training_config.do_eval,
        'eval_steps': training_config.eval_steps,
        'seed': training_config.seed,
        
        # Core training parameters
        'num_train_epochs': training_config.num_train_epochs,
        'max_steps': training_config.max_steps,
        'per_device_train_batch_size': training_config.per_device_train_batch_size,
        'per_device_eval_batch_size': training_config.per_device_eval_batch_size,
        'logging_dir': training_config.logging_dir,
        'resume_from_checkpoint': training_config.resume_from_checkpoint
    }
    
    # Combine all parameters
    all_params = {**task_params, **model_params, **training_params}
    
    # Set attributes on args object if provided
    if args is not None:
        for key, value in all_params.items():
            setattr(args, key, value)
    
    return all_params


def calculate_image_sizes(camera_names: list, image_size_primary: str, image_size_wrist: str) -> list:
    """Calculate image sizes list for backward compatibility."""
    return [image_size_primary if 'primary' in cam else image_size_wrist for cam in camera_names]
