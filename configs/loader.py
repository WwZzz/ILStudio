import os
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from .utils import resolve_yaml, parse_overrides, apply_overrides_to_mapping, apply_overrides_to_object
from .training.loader import load_training_config
from data_utils.utils import _convert_to_type
from types import SimpleNamespace


class ConfigLoader:
    """Unified loader for training, task, policy, robot, teleop configs with CLI overrides."""

    def __init__(self, args=None, unknown_args=None):
        self.args = args
        self.unknown_args = unknown_args
        # Accept either a list of unknown args or a precomputed overrides dict
        if isinstance(self.unknown_args, dict):
            self._overrides = self.unknown_args
        else:
            self._overrides = parse_overrides(self.unknown_args or [])

    def get_overrides(self, category: str) -> Dict[str, Any]:
        return self._overrides.get(category, {})

    def _base_dir(self, category: str) -> str:
        base = Path(__file__).resolve().parent
        return str(base / category)

    def _resolve(self, category: str, name_or_path: str) -> str:
        try:
            return resolve_yaml(name_or_path, self._base_dir(category))
        except FileNotFoundError:
            return name_or_path

    def load_yaml_config(self, category: str, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        path = self._resolve(category, name_or_path)
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        apply_overrides_to_mapping(cfg, self.get_overrides(category), _convert_to_type)
        return cfg, path

    def load_task(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        return self.load_yaml_config('task', name_or_path)

    def load_policy(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        cfg, path = self.load_yaml_config('policy', name_or_path)
        
        # Flatten model_args to top level for easier command line access
        # This allows --policy.camera_names, --policy.chunk_size, etc.
        if 'model_args' in cfg and isinstance(cfg['model_args'], dict):
            model_args = cfg['model_args']
            # Create a flattened copy while preserving the original model_args
            flattened_cfg = cfg.copy()
            for key, value in model_args.items():
                # Only add to top level if not already present at top level
                if key not in flattened_cfg:
                    flattened_cfg[key] = value
            cfg = flattened_cfg
        
        return cfg, path

    def load_robot(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        return self.load_yaml_config('robot', name_or_path)

    def load_teleop(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        return self.load_yaml_config('teleop', name_or_path)

    def load_env(self, name_or_path: str) -> Tuple[Any, str]:
        """Load env config and return a namespace for attribute-style access.
        Expects a key 'type' in the YAML to indicate which benchmark env to load (e.g., 'aloha', 'libero', 'robomimic').
        """
        cfg, path = self.load_yaml_config('env', name_or_path)
        # recursive dict -> namespace
        def to_ns(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [to_ns(x) for x in d]
            return d
        return to_ns(cfg), path

    def load_training(self, name_or_path: str, hyper_args=None):
        """Return (training_config_obj, training_args_obj, resolved_path)."""
        path = self._resolve('training', name_or_path)
        training_config = load_training_config(path)
        # apply overrides to object
        apply_overrides_to_object(training_config, self.get_overrides('training'), _convert_to_type)
        if hyper_args is None:
            hyper_args = self.args
        training_args = training_config.to_training_arguments(hyper_args)
        return training_config, training_args, path

    # ===== Parameter merging (moved from parameter_merger.py) =====
    @staticmethod
    def calculate_image_sizes(camera_names: list, image_size_primary: str, image_size_wrist: str) -> list:
        return [image_size_primary if 'primary' in cam else image_size_wrist for cam in camera_names]

    @staticmethod
    def merge_all_parameters(task_config: Dict[str, Any], policy_config: Dict[str, Any], training_config: Any, args: Optional[Any] = None) -> Dict[str, Any]:
        task_params = {
            'action_dim': task_config.get('action_dim', 7),
            'state_dim': task_config.get('state_dim', 7),
            'camera_names': task_config.get('camera_names', []),
            'image_size_primary': task_config.get('image_size_primary', '(256, 256)'),
            'image_size_wrist': task_config.get('image_size_wrist', '(256, 256)'),
            'use_reasoning': task_config.get('use_reasoning', False),
            'use_prev_subtask': task_config.get('use_prev_subtask', False)
        }

        # Extract model_args and pretrained_config
        model_args = policy_config.get('model_args', {})
        pretrained_config = policy_config.get('pretrained_config', {})

        # Dynamically extract all model parameters from policy config
        # This includes both model_args and any flattened parameters from command line overrides
        model_params = {}
        
        # First add pretrained_config parameters
        if pretrained_config:
            model_params.update(pretrained_config)
        
        # Then add all model_args parameters
        if model_args:
            model_params.update(model_args)
        
        # Finally add any top-level parameters that were flattened (from command line overrides)
        # Skip known non-model parameters like 'name', 'module_path', 'model_args', 'pretrained_config'
        reserved_keys = {'name', 'module_path', 'model_args', 'pretrained_config', 'config_class', 'model_class', 'data_processor', 'data_collator', 'trainer_class'}
        for key, value in policy_config.items():
            if key not in reserved_keys and key not in model_params:
                model_params[key] = value
            elif key not in reserved_keys and key in model_params:
                # Top-level overrides win (command line overrides)
                model_params[key] = value

        training_params = {
            'preload_data': training_config.preload_data,
            'logging_strategy': training_config.logging_strategy,
            'logging_steps': training_config.logging_steps,
            'report_to': training_config.report_to,
            'save_strategy': training_config.save_strategy,
            'save_steps': training_config.save_steps,
            'save_total_limit': training_config.save_total_limit,
            'dataloader_num_workers': training_config.dataloader_num_workers,
            'dataloader_pin_memory': training_config.dataloader_pin_memory,
            'remove_unused_columns': training_config.remove_unused_columns,
            'do_eval': training_config.do_eval,
            'eval_steps': training_config.eval_steps,
            'seed': training_config.seed,
            'num_train_epochs': training_config.num_train_epochs,
            'max_steps': training_config.max_steps,
            'per_device_train_batch_size': training_config.per_device_train_batch_size,
            'per_device_eval_batch_size': training_config.per_device_eval_batch_size,
            'logging_dir': training_config.logging_dir,
            'resume_from_checkpoint': training_config.resume_from_checkpoint
        }

        cfg_params = policy_config.get('config_params', {}) if isinstance(policy_config, dict) else {}
        # Check top-level first (command line overrides), then model_args, then config_params, then task_config
        preferred_chunk_size = (policy_config.get('chunk_size') or 
                              model_args.get('chunk_size') or 
                              cfg_params.get('chunk_size') or 
                              task_config.get('chunk_size', 16))
        preferred_action_norm = (policy_config.get('action_normalize') or 
                               model_args.get('action_normalize') or 
                               cfg_params.get('action_normalize') or 
                               task_config.get('action_normalize', 'minmax'))
        preferred_state_norm = (policy_config.get('state_normalize') or 
                              model_args.get('state_normalize') or 
                              cfg_params.get('state_normalize') or 
                              task_config.get('state_normalize', 'minmax'))
        preferred_camera_names = policy_config.get('camera_names', None) 
        if preferred_camera_names is None:
            preferred_camera_names = model_args.get('camera_names', None) 
            if preferred_camera_names is None:
                preferred_camera_names =task_config.get('camera_names', [])

        all_params = {**task_params, **model_params, **training_params}
        all_params.update({
            'chunk_size': preferred_chunk_size,
            'action_normalize': preferred_action_norm,
            'state_normalize': preferred_state_norm,
            'camera_names': preferred_camera_names,  # Allow policy to override camera_names
        })
        
        # Remove None values to avoid overriding existing values with None
        all_params = {k: v for k, v in all_params.items() if v is not None}

        if args is not None:
            for key, value in all_params.items():
                setattr(args, key, value)
        args.image_sizes = self.calculate_image_sizes(args.camera_names, args.image_size_primary, args.image_size_wrist)
        return all_params


