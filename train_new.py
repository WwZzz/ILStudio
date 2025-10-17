import os
import argparse
import yaml
import json
import transformers
import policy.utils as ml_utils
from data_utils.utils import set_seed, WrappedDataset, load_data, get_dataloader
from configs.loader import ConfigLoader
from configs.training.loader import load_training_config
from policy.policy_loader import (
    get_policy_data_processor,
    get_policy_data_collator,
    get_policy_trainer_class,
    load_policy_model_for_training,
)
from accelerate import Accelerator
from policy.trainer import BaseTrainer


# Removed HyperArguments dataclass - using simple argparse instead

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
        training_args: TrainingArguments object
    """
    parser = argparse.ArgumentParser(description='Train a policy model')
    
    # Essential arguments
    parser.add_argument('-p', '--policy', type=str, default='act',
                       help='Policy config (name under configs/policy or absolute path to yaml)')
    parser.add_argument('-t', '--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task config (name under configs/task or absolute path to yaml)')
    parser.add_argument('-c', '--training_config', type=str, default='default',
                       help='Training config (name under configs/training or absolute path to yaml)')
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/training_output',
                       help='Output directory for checkpoints')
    
    # Parse arguments (allow unknown for dotted overrides)
    args, unknown = parser.parse_known_args()
    
    # Load training configuration via unified config loader
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
    
    # Use the existing ConfigLoader instance that was created in parse_param()
    # Don't create new instances - reuse the one with correct overrides
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'config_overrides', {}))
    
    # Load all configurations using the same loader
    task_config, task_cfg_path = cfg_loader.load_task(args.task)
    policy_config, policy_cfg_path = cfg_loader.load_policy(args.policy)
    
    # Load training config using the same loader instead of separate function
    training_config, _, training_cfg_path = cfg_loader.load_training(getattr(args, 'training_cfg_path', args.training_config), hyper_args=args)
    
    # Merge all parameters using unified loader
    ConfigLoader.merge_all_parameters(task_config, policy_config, training_config, args)
    
    # Save policy metadata to output dir
    metadata_path = os.path.join(training_args.output_dir, 'policy_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
                'policy_module': policy_config['module_path'],
                'policy_name': policy_config['name'],
            }, f, indent=2)
    
    # Load model 
    print(f"Loading policy config: {policy_cfg_path}")
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
    
    # Create data loader
    train_loader, eval_loader = get_dataloader(train_dataset, val_dataset, data_processor, data_collator, args) 
    
    # Get Trainer
    train_class = get_policy_trainer_class(policy_cfg_path) or BaseTrainer
    trainer = train_class(
        args=training_args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # Save model
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    args, training_args = parse_param()
    os.makedirs(training_args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(training_args.output_dir, ckpt_name) for ckpt_name in os.listdir(training_args.output_dir) if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(training_args.output_dir, ckpt_name))]
    if len(all_ckpts)==0: 
        training_args.resume_from_checkpoint = None
    main(args, training_args)
