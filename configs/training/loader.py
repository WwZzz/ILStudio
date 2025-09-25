"""
Training Configuration Loader

This module provides utilities to load training configurations from YAML files
and convert them to transformers.TrainingArguments.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import transformers
from ..utils import resolve_yaml


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML file - only stable process parameters."""
    
    # Data loading (training process related)
    preload_data: bool = False
    
    # =============================================================================
    # OPTIMIZER CONFIGURATION
    # =============================================================================
    # Core optimizer settings that are typically set once per training setup
    
    # Learning rate and scheduling
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "constant"
    
    # Optimizer settings
    optim: str = "adamw_torch"
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Note: Model architecture, LoRA, and quantization settings are now in configs/model/default.yaml
    
    # =============================================================================
    # LOGGING AND SAVING CONFIGURATION
    # =============================================================================
    # Logging and saving settings that are typically set once
    
    # Logging settings
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: str = "tensorboard"
    
    # Saving settings
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 50
    
    # =============================================================================
    # DATA PROCESSING CONFIGURATION (TrainingArguments)
    # =============================================================================
    # Data processing settings for the training process
    
    # DataLoader settings
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    
    # Evaluation settings
    do_eval: bool = False
    eval_steps: int = 200
    
    # System settings
    seed: int = 0
    
    # =============================================================================
    # CORE TRAINING PARAMETERS
    # =============================================================================
    # Basic training settings that are typically set once per experiment
    
    # Training duration
    num_train_epochs: int = 3
    max_steps: int = 5000
    
    # Batch sizes (typically determined by hardware)
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    
    # Logging directory (often derived from output_dir)
    logging_dir: str = "./logs"
    
    # Checkpoint resuming (often handled automatically)
    resume_from_checkpoint: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load training configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create instance with default values
        instance = cls()
        
        # Update with values from YAML
        for key, value in config_data.items():
            if hasattr(instance, key):
                # Handle scientific notation strings that should be floats
                if key in ['adam_epsilon', 'learning_rate', 'weight_decay', 'warmup_ratio'] and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"Warning: Could not convert {key}='{value}' to float, using default")
                        continue
                setattr(instance, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' in {yaml_path}")
        
        return instance
    
    def to_training_arguments(self, hyper_args, **overrides) -> transformers.TrainingArguments:
        """Convert to transformers.TrainingArguments using HyperArguments and optional overrides."""
        # Get core training parameters from HyperArguments
        config_dict = {
            'output_dir': hyper_args.output_dir,
            # Core training parameters now come from training config
            'num_train_epochs': self.num_train_epochs,
            'max_steps': self.max_steps,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'per_device_eval_batch_size': self.per_device_eval_batch_size,
            # Optimizer parameters now come from training config
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'lr_scheduler_type': self.lr_scheduler_type,
            'optim': self.optim,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_epsilon': self.adam_epsilon,
            'logging_dir': self.logging_dir,
            # Logging and saving parameters now come from training config
            'logging_strategy': self.logging_strategy,
            'logging_steps': self.logging_steps,
            'report_to': self.report_to,
            'save_strategy': self.save_strategy,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            # Data processing parameters now come from training config
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'remove_unused_columns': self.remove_unused_columns,
            'do_eval': self.do_eval,
            'eval_steps': self.eval_steps,
            'seed': self.seed,
        }
        
        # Apply overrides
        config_dict.update(overrides)
        
        return transformers.TrainingArguments(**config_dict)


def load_training_config(config_path: str = "configs/training/default.yaml") -> TrainingConfig:
    """Load training configuration from YAML file. Accepts name or path."""
    base_dir = os.path.join(Path(__file__).resolve().parent)
    try:
        resolved = resolve_yaml(config_path, base_dir)
    except FileNotFoundError:
        resolved = config_path
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Training configuration not found: {resolved}")
    return TrainingConfig.from_yaml(resolved)


def create_training_arguments(config_path: str = "configs/training/default.yaml", hyper_args=None, **overrides) -> transformers.TrainingArguments:
    """Load training configuration and convert to TrainingArguments with overrides."""
    config = load_training_config(config_path)
    if hyper_args is None:
        raise ValueError("hyper_args must be provided to create TrainingArguments")
    return config.to_training_arguments(hyper_args, **overrides)
