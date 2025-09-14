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



# Removed HyperArguments dataclass - using simple argparse instead

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
    parser.add_argument('--policy_config', type=str, default='configs/policy/act.yaml',
                       help='Policy config file path')
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube_scripted',
                       help='Task name')
    parser.add_argument('--training_config', type=str, default='configs/training/default.yaml',
                       help='Training config file path')
    parser.add_argument('--output_dir', type=str, default='ckpt/training_output',
                       help='Output directory for checkpoints')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load training configuration
    from configs.training.loader import load_training_config, create_training_arguments
    
    print(f"Loading training config: {args.training_config}")
    training_config = load_training_config(args.training_config)
    
    # Create TrainingArguments
    training_args = create_training_arguments(args.training_config, args)
    
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
    # 初始化任务信息
    set_seed(1)
    task_config = load_task_config(args.task_name)
    
    # Load configurations
    from configs.training.loader import load_training_config
    from configs.parameter_merger import merge_all_parameters, calculate_image_sizes
    import yaml
    
    training_config = load_training_config(args.training_config)
    
    # Load policy config
    with open(args.policy_config, 'r') as f:
        policy_config = yaml.safe_load(f)
    
    # Merge all parameters using the parameter merger utility
    merge_all_parameters(task_config, policy_config, training_config, args)
    
    # Calculate image sizes for backward compatibility
    args.image_sizes = calculate_image_sizes(args.camera_names, args.image_size_primary, args.image_size_wrist)
    
    # 加载模型 - 使用新的策略加载器
    from policy.policy_loader import load_policy_model, get_policy_data_processor, get_policy_data_collator, get_policy_trainer_class
    
    # 使用新的策略加载器加载模型，传入task_config
    print(f"Loading policy config: {args.policy_config}")
    from policy.policy_loader import load_policy_model_for_training
    model_components = load_policy_model_for_training(args.policy_config, args, task_config)
    model = model_components['model']
    config = model_components.get('config', None)
    if config:
        print(f"Loaded config from YAML: {type(config).__name__}")
    ml_utils.print_model_trainable_information(model)
    
    # 加载数据集
    train_dataset, val_dataset = load_data(args, task_config)
    
    # 包装数据集
    data_processor = get_policy_data_processor(args.policy_config, args, model_components)
    data_collator = get_policy_data_collator(args.policy_config, args, model_components)
    
    data_module = dict(
        train_dataset=WrappedDataset(train_dataset, data_processor) if data_processor is not None else train_dataset,
        eval_dataset=WrappedDataset(val_dataset, data_processor) if data_processor is not None and val_dataset is not None else val_dataset,
        data_collator=data_collator,
    )
    
    # 获取 Trainer
    train_class = get_policy_trainer_class(args.policy_config) or transformers.trainer.Trainer
    trainer = train_class(
        args=training_args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        **data_module
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # 保存模型
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
