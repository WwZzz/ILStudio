import os
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
import importlib
import policy.utils as ml_utils
from configs.task.loader import load_task_config
from data_utils.utils import set_seed, WrappedDataset, load_data, _convert_to_type
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
from pathlib import Path



# Removed HyperArguments dataclass - using simple argparse instead

def _set_nested(obj, keys, value):
    cur = obj
    for k in keys[:-1]:
        if isinstance(cur, dict):
            if k not in cur or not isinstance(cur[k], (dict,)):
                cur[k] = {}
            cur = cur[k]
        else:
            if not hasattr(cur, k) or not isinstance(getattr(cur, k), (dict,)):
                setattr(cur, k, {})
            cur = getattr(cur, k)
    last = keys[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)

def _parse_overrides(unknown_args):
    overrides = { 'task': {}, 'training': {}, 'policy': {} }
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith('--'):
            i += 1
            continue
        key = token[2:]
        value = None
        if '=' in key:
            key, value = key.split('=', 1)
        else:
            if i + 1 < len(unknown_args) and not unknown_args[i+1].startswith('--'):
                value = unknown_args[i+1]
                i += 1
        if key.startswith('task.') or key.startswith('training.') or key.startswith('policy.'):
            root, subpath = key.split('.', 1)
            overrides[root][subpath] = value
        i += 1
    return overrides

def _apply_overrides_to_mapping(mapping_obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        val = caster(raw)
        keys = dotted.split('.')
        _set_nested(mapping_obj, keys, val)

def _apply_overrides_to_object(obj, flat_overrides, caster):
    for dotted, raw in flat_overrides.items():
        if raw is None:
            continue
        val = caster(raw)
        keys = dotted.split('.')
        _set_nested(obj, keys, val)

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
        training_args: TrainingArguments object
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a policy model')
    
    # Essential arguments
    parser.add_argument('--policy', type=str, default='act',
                       help='Policy config (name under configs/policy or absolute path to yaml)')
    parser.add_argument('--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task config (name under configs/task or absolute path to yaml)')
    parser.add_argument('--training_config', type=str, default='default',
                       help='Training config (name under configs/training or absolute path to yaml)')
    parser.add_argument('--output_dir', type=str, default='ckpt/training_output',
                       help='Output directory for checkpoints')
    
    # Parse arguments (allow unknown for dotted overrides)
    args, unknown = parser.parse_known_args()
    
    # Load training configuration
    # Unified config loader
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
    training_config, training_args, training_cfg_path = cfg_loader.load_training(args.training_config, hyper_args=args)
    setattr(args, 'config_overrides', cfg_loader._overrides)
    setattr(args, 'training_cfg_path', training_cfg_path)
    return args, training_args

def main(args, training_args):
    """
    Main training function for the VLA (Vision-Language-Action) model.

    Args:
        args (HyperArguments): Training hyperparameters and settings
        training_args (transformers.TrainingArguments): Training arguments from config

    Returns:
        None. The trained model and statistics are saved to the output directory
        specified in training_args.
    """
    # Init task config
    set_seed(1)
    # Resolve and load task config via ConfigLoader (with overrides)
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'config_overrides', {}))
    task_config, task_cfg_path = cfg_loader.load_task(args.task)
    # Apply task overrides if any
    overrides = getattr(args, 'config_overrides', {'task': {}})
    from data_utils.utils import _convert_to_type
    _apply_overrides_to_mapping(task_config, overrides.get('task', {}), _convert_to_type)
    
    # Load configurations
    from configs.training.loader import load_training_config
    from configs.loader import ConfigLoader
    import yaml
    
    training_config = load_training_config(getattr(args, 'training_cfg_path', args.training_config))
    
    # Load policy config
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'config_overrides', {}))
    policy_config, policy_cfg_path = cfg_loader.load_policy(args.policy)
    
    # Merge all parameters using unified loader
    ConfigLoader.merge_all_parameters(task_config, policy_config, training_config, args)
    
    # Calculate image sizes for backward compatibility
    args.image_sizes = ConfigLoader.calculate_image_sizes(args.camera_names, args.image_size_primary, args.image_size_wrist)
    
    # Load model - using new policy loader
    from policy.policy_loader import load_policy_model, get_policy_data_processor, get_policy_data_collator, get_policy_trainer_class
    
    # Load model - using new policy loader, pass task_config
    print(f"Loading policy config: {policy_cfg_path}")
    from policy.policy_loader import load_policy_model_for_training
    model_components = load_policy_model_for_training(policy_cfg_path, args, task_config)
    model = model_components['model']
    config = model_components.get('config', None)
    if config:
        print(f"Loaded config from YAML: {type(config).__name__}")
    ml_utils.print_model_trainable_information(model)
    
    # Load dataset
    train_dataset, val_dataset = load_data(args, task_config)
    
    # Wrap dataset
    data_processor = get_policy_data_processor(policy_cfg_path, args, model_components)
    data_collator = get_policy_data_collator(policy_cfg_path, args, model_components)
    
    data_module = dict(
        train_dataset=WrappedDataset(train_dataset, data_processor) if data_processor is not None else train_dataset,
        eval_dataset=WrappedDataset(val_dataset, data_processor) if data_processor is not None and val_dataset is not None else val_dataset,
        data_collator=data_collator,
    )
    
    # Get Trainer
    train_class = get_policy_trainer_class(policy_cfg_path) or transformers.trainer.Trainer
    trainer = train_class(
        args=training_args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        **data_module
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # Save model
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        
        # Save policy module information for direct loading
        policy_metadata = {
            'policy_module': policy_config['module_path'],
            'policy_name': policy_config['name'],
        }
        import json
        metadata_path = os.path.join(training_args.output_dir, 'policy_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(policy_metadata, f, indent=2)
        print(f"Saved policy metadata to {metadata_path}")

if __name__ == '__main__':
    args, training_args = parse_param()
    os.makedirs(training_args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(training_args.output_dir, ckpt_name) for ckpt_name in os.listdir(training_args.output_dir) if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(training_args.output_dir, ckpt_name))]
    if len(all_ckpts)==0: 
        training_args.resume_from_checkpoint = None
    main(args, training_args)
