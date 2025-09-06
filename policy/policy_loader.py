"""
Policy Configuration Loader

This module provides utilities to load model configurations from YAML files
and dynamically import and instantiate model modules.
"""

import yaml
import importlib
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PolicyConfig:
    """Configuration for a policy model."""
    name: str
    module_path: str
    config_class: str
    model_class: str
    pretrained_config: Optional[Dict[str, Any]] = None
    model_args: Optional[Dict[str, Any]] = None
    config_params: Optional[Dict[str, Any]] = None  # Parameters for config class initialization
    data_processor: Optional[str] = None
    data_collator: Optional[str] = None
    trainer_class: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PolicyConfig':
        """Load policy configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            name=config_data['name'],
            module_path=config_data['module_path'],
            config_class=config_data['config_class'],
            model_class=config_data['model_class'],
            pretrained_config=config_data.get('pretrained_config'),
            model_args=config_data.get('model_args', {}),
            config_params=config_data.get('config_params', {}),
            data_processor=config_data.get('data_processor'),
            data_collator=config_data.get('data_collator'),
            trainer_class=config_data.get('trainer_class')
        )


class PolicyLoader:
    """Dynamic policy loader that loads models based on YAML configurations."""
    
    def __init__(self, policy_dir: str = "configs/policy"):
        self.policy_dir = Path(policy_dir)
        self._loaded_modules = {}
    
    def load_policy_config(self, policy_name_or_path: str) -> PolicyConfig:
        """Load policy configuration from YAML file."""
        # Check if it's a full path or just a policy name
        if policy_name_or_path.endswith('.yaml') or '/' in policy_name_or_path:
            # It's a full path
            yaml_path = Path(policy_name_or_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Policy configuration not found: {yaml_path}")
        else:
            # It's just a policy name, construct the path
            yaml_path = self.policy_dir / f"{policy_name_or_path}.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"Policy configuration not found: {yaml_path}")
        
        return PolicyConfig.from_yaml(str(yaml_path))
    
    def load_model_module(self, policy_config: PolicyConfig):
        """Dynamically load the model module."""
        if policy_config.module_path in self._loaded_modules:
            return self._loaded_modules[policy_config.module_path]
        
        try:
            module = importlib.import_module(policy_config.module_path)
            self._loaded_modules[policy_config.module_path] = module
            return module
        except ImportError as e:
            raise ImportError(f"Failed to import module {policy_config.module_path}: {e}")
    
    def create_config_instance(self, policy_config: PolicyConfig, args=None, task_config=None, phase='training'):
        """Create a config instance using parameters from YAML, task config, and args."""
        # Load the module to get access to the config class
        model_module = self.load_model_module(policy_config)
        
        # Get the config class
        if not hasattr(model_module, policy_config.config_class):
            raise AttributeError(f"Module {policy_config.module_path} does not have config class {policy_config.config_class}")
        
        config_class = getattr(model_module, policy_config.config_class)
        
        # Prepare parameters for config initialization
        config_kwargs = {}
        
        # Start with config_params from YAML
        if policy_config.config_params:
            config_kwargs.update(policy_config.config_params)
        
        # For training phase: Override with task config parameters if provided
        if phase == 'training' and task_config:
            # Map task config parameters to config parameters
            task_mapping = {
                'action_dim': 'action_dim',
                'state_dim': 'state_dim',
                'chunk_size': 'chunk_size',
                'camera_names': 'camera_names',
                'image_size_primary': 'image_size_primary',
                'image_size_wrist': 'image_size_wrist',
                'action_normalize': 'action_normalize',
                'state_normalize': 'state_normalize'
            }
            
            for task_name, config_name in task_mapping.items():
                if task_name in task_config:
                    value = task_config[task_name]
                    # Special handling for camera_names - convert string to list if needed
                    if config_name == 'camera_names' and isinstance(value, str):
                        import ast
                        value = ast.literal_eval(value)
                    config_kwargs[config_name] = value
            
            # Special handling for image_sizes - convert image_size_primary and image_size_wrist to image_sizes list
            if 'image_size_primary' in task_config and 'image_size_wrist' in task_config and 'camera_names' in task_config:
                image_sizes = []
                for cam_name in task_config['camera_names']:
                    if 'primary' in cam_name:
                        image_sizes.append(task_config['image_size_primary'])
                    else:
                        image_sizes.append(task_config['image_size_wrist'])
                config_kwargs['image_sizes'] = image_sizes
            
            # Special handling for prediction_horizon - map chunk_size to prediction_horizon
            if 'chunk_size' in task_config:
                config_kwargs['prediction_horizon'] = task_config['chunk_size']
        
        # For evaluation phase: Load from saved model config if available
        elif phase == 'evaluation' and hasattr(args, 'model_name_or_path') and args.model_name_or_path:
            saved_config_path = os.path.join(args.model_name_or_path, 'config.json')
            if os.path.exists(saved_config_path):
                try:
                    import json
                    with open(saved_config_path, 'r') as f:
                        saved_config = json.load(f)
                    
                    # Map saved config parameters to config parameters
                    saved_mapping = {
                        'action_dim': 'action_dim',
                        'state_dim': 'state_dim',
                        'chunk_size': 'chunk_size',
                        'camera_names': 'camera_names',
                        'image_size_primary': 'image_size_primary',
                        'image_size_wrist': 'image_size_wrist',
                        'action_normalize': 'action_normalize',
                        'state_normalize': 'state_normalize',
                        'lr_backbone': 'lr_backbone',
                        'kl_weight': 'kl_weight',
                        'backbone': 'backbone',
                        'hidden_dim': 'hidden_dim',
                        'enc_layers': 'enc_layers',
                        'dec_layers': 'dec_layers',
                        'dim_feedforward': 'dim_feedforward',
                        'dropout': 'dropout',
                        'nheads': 'nheads',
                        'pre_norm': 'pre_norm',
                        'masks': 'masks',
                        'dilation': 'dilation',
                        'position_embedding': 'position_embedding'
                    }
                    
                    for saved_name, config_name in saved_mapping.items():
                        if saved_name in saved_config:
                            value = saved_config[saved_name]
                            # Special handling for camera_names - convert string to list if needed
                            if config_name == 'camera_names' and isinstance(value, str):
                                import ast
                                value = ast.literal_eval(value)
                            config_kwargs[config_name] = value
                    
                    # Also set parameters in args for backward compatibility
                    eval_params = {
                        'action_dim': saved_config.get('action_dim', 7),
                        'state_dim': saved_config.get('state_dim', 7),
                        'chunk_size': saved_config.get('chunk_size', 50),
                        'camera_names': saved_config.get('camera_names', ['primary']),
                        'image_size_primary': saved_config.get('image_size_primary', "(640, 480)"),
                        'image_size_wrist': saved_config.get('image_size_wrist', "(640, 480)"),
                        'action_normalize': saved_config.get('action_normalize', 'minmax'),
                        'state_normalize': saved_config.get('state_normalize', 'minmax')
                    }
                    
                    # Set evaluation parameters in args for backward compatibility
                    for key, value in eval_params.items():
                        setattr(args, key, value)
                    
                    print(f"Loaded parameters from saved config: {saved_config_path}")
                except Exception as e:
                    print(f"Warning: Could not load saved config from {saved_config_path}: {e}")
                    print("Falling back to YAML config parameters")
        
        # Override with args if provided (highest priority)
        if args:
            # Map common args to config parameters
            arg_mapping = {
                'state_dim': 'state_dim',
                'action_dim': 'action_dim', 
                'chunk_size': 'chunk_size',
                'camera_names': 'camera_names',
                'image_size_primary': 'image_size_primary',
                'image_size_wrist': 'image_size_wrist',
                'action_normalize': 'action_normalize',
                'state_normalize': 'state_normalize',
                'lr_backbone': 'lr_backbone',
                'kl_weight': 'kl_weight',
                'backbone': 'backbone',
                'hidden_dim': 'hidden_dim',
                'enc_layers': 'enc_layers',
                'dec_layers': 'dec_layers',
                'dim_feedforward': 'dim_feedforward',
                'dropout': 'dropout',
                'nheads': 'nheads',
                'pre_norm': 'pre_norm',
                'masks': 'masks',
                'dilation': 'dilation',
                'position_embedding': 'position_embedding'
            }
            
            for arg_name, config_name in arg_mapping.items():
                if hasattr(args, arg_name):
                    value = getattr(args, arg_name)
                    # Special handling for camera_names - convert string to list
                    if config_name == 'camera_names' and isinstance(value, str):
                        import ast
                        value = ast.literal_eval(value)
                    config_kwargs[config_name] = value
        
        # Create and return the config instance
        try:
            config_instance = config_class(**config_kwargs)
            return config_instance
        except Exception as e:
            raise ValueError(f"Failed to create config instance with parameters {config_kwargs}: {e}")
    
    def load_model_with_config(self, policy_name_or_path: str, args, task_config=None, phase='training') -> Dict[str, Any]:
        """Load model using policy configuration with custom config instance."""
        # Load policy configuration
        policy_config = self.load_policy_config(policy_name_or_path)
        
        # Load model module
        model_module = self.load_model_module(policy_config)
        
        # Create config instance from YAML parameters, task config, and args
        config_instance = self.create_config_instance(policy_config, args, task_config, phase)
        
        # Get the model class
        if not hasattr(model_module, policy_config.model_class):
            raise AttributeError(f"Module {policy_config.module_path} does not have model class {policy_config.model_class}")
        
        model_class = getattr(model_module, policy_config.model_class)
        
        # Create model instance with the config
        try:
            model = model_class(config=config_instance)
            return {'model': model, 'config': config_instance}
        except Exception as e:
            raise ValueError(f"Failed to create model instance: {e}")
    
    def load_model_for_training(self, policy_name_or_path: str, args, task_config=None) -> Dict[str, Any]:
        """Load model for training phase - uses task config parameters."""
        return self.load_model_with_config(policy_name_or_path, args, task_config, phase='training')
    
    def load_model_for_evaluation(self, policy_name_or_path: str, args) -> Dict[str, Any]:
        """Load model for evaluation phase - uses saved model config parameters."""
        return self.load_model_with_config(policy_name_or_path, args, task_config=None, phase='evaluation')
    
    def load_model(self, policy_name_or_path: str, args) -> Dict[str, Any]:
        """Load model using policy configuration."""
        # Load policy configuration
        policy_config = self.load_policy_config(policy_name_or_path)
        
        # Load model module
        model_module = self.load_model_module(policy_config)
        
        # Verify required methods exist
        if not hasattr(model_module, 'load_model'):
            raise AttributeError(f"Module {policy_config.module_path} must provide 'load_model' function")
        
        # Call the module's load_model function
        model_components = model_module.load_model(args)
        
        return model_components
    
    def get_data_processor(self, policy_name_or_path: str, dataset, args, model_components):
        """Get data processor for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        if policy_config.data_processor and hasattr(model_module, policy_config.data_processor):
            processor_func = getattr(model_module, policy_config.data_processor)
            return processor_func(dataset, args, model_components)
        
        return None
    
    def get_data_collator(self, policy_name_or_path: str, args, model_components):
        """Get data collator for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        if policy_config.data_collator and hasattr(model_module, policy_config.data_collator):
            collator_func = getattr(model_module, policy_config.data_collator)
            return collator_func(args, model_components)
        
        return None
    
    def get_trainer_class(self, policy_name_or_path: str):
        """Get trainer class for the policy."""
        policy_config = self.load_policy_config(policy_name_or_path)
        model_module = self.load_model_module(policy_config)
        
        if policy_config.trainer_class and hasattr(model_module, policy_config.trainer_class):
            return getattr(model_module, policy_config.trainer_class)
        
        return None
    
    def list_available_policies(self) -> list:
        """List all available policy configurations."""
        yaml_files = list(self.policy_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]


# Global policy loader instance
policy_loader = PolicyLoader()


def load_policy_model(policy_name_or_path: str, args) -> Dict[str, Any]:
    """Convenience function to load a policy model."""
    return policy_loader.load_model(policy_name_or_path, args)


def load_policy_model_with_config(policy_name_or_path: str, args, task_config=None, phase='training') -> Dict[str, Any]:
    """Convenience function to load a policy model with custom config from YAML."""
    return policy_loader.load_model_with_config(policy_name_or_path, args, task_config, phase)

def load_policy_model_for_training(policy_name_or_path: str, args, task_config=None) -> Dict[str, Any]:
    """Convenience function to load a policy model for training - uses task config parameters."""
    return policy_loader.load_model_for_training(policy_name_or_path, args, task_config)

def load_policy_model_for_evaluation(policy_name_or_path: str, args) -> Dict[str, Any]:
    """Convenience function to load a policy model for evaluation - uses saved model config parameters."""
    return policy_loader.load_model_for_evaluation(policy_name_or_path, args)


def get_policy_data_processor(policy_name_or_path: str, dataset, args, model_components):
    """Convenience function to get data processor."""
    return policy_loader.get_data_processor(policy_name_or_path, dataset, args, model_components)


def get_policy_data_collator(policy_name_or_path: str, args, model_components):
    """Convenience function to get data collator."""
    return policy_loader.get_data_collator(policy_name_or_path, args, model_components)


def get_policy_trainer_class(policy_name_or_path: str):
    """Convenience function to get trainer class."""
    return policy_loader.get_trainer_class(policy_name_or_path)
