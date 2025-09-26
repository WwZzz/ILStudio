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
        return self.load_yaml_config('policy', name_or_path)

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
            'camera_names': task_config.get('camera_names', ['primary']),
            'image_size_primary': task_config.get('image_size_primary', '(256, 256)'),
            'image_size_wrist': task_config.get('image_size_wrist', '(256, 256)'),
            'use_reasoning': task_config.get('use_reasoning', False),
            'use_prev_subtask': task_config.get('use_prev_subtask', False)
        }

        model_args = policy_config.get('model_args', {})
        pretrained_config = policy_config.get('pretrained_config', {})

        model_params = {
            'lora_enable': model_args.get('lora_enable', False),
            'lora_module': model_args.get('lora_module', 'all'),
            'lora_task_type': model_args.get('lora_task_type', 'CAUSAL_LM'),
            'lora_r': model_args.get('lora_r', 16),
            'lora_alpha': model_args.get('lora_alpha', 32),
            'lora_dropout': model_args.get('lora_dropout', 0.1),
            'lora_weight_path': model_args.get('lora_weight_path', None),
            'lora_bias': model_args.get('lora_bias', 'none'),
            'lora_lr': model_args.get('lora_lr', 0.0002),
            'use_quantization': model_args.get('use_quantization', False),
            'bits': model_args.get('bits', 4),
            'double_quant': model_args.get('double_quant', True),
            'quant_type': model_args.get('quant_type', 'nf4'),
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
            'lazy_preprocess': model_args.get('lazy_preprocess', False),
            'select_seg_token_mask': model_args.get('select_seg_token_mask', False),
            'is_multimodal': model_args.get('is_multimodal', True),
            'image_aspect_ratio': model_args.get('image_aspect_ratio', 'square'),
            'skip_mirrored_data': model_args.get('skip_mirrored_data', False),
            'history_images_length': model_args.get('history_images_length', 1),
        }

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
        model_args = policy_config.get('model_args', {}) if isinstance(policy_config, dict) else {}
        preferred_chunk_size = cfg_params.get('chunk_size', model_args.get('chunk_size', task_config.get('chunk_size', 16)))
        preferred_action_norm = cfg_params.get('action_normalize', model_args.get('action_normalize', task_config.get('action_normalize', 'minmax')))
        preferred_state_norm = cfg_params.get('state_normalize', model_args.get('state_normalize', task_config.get('state_normalize', 'minmax')))

        all_params = {**task_params, **model_params, **training_params}
        all_params.update({
            'chunk_size': preferred_chunk_size,
            'action_normalize': preferred_action_norm,
            'state_normalize': preferred_state_norm,
        })

        if args is not None:
            for key, value in all_params.items():
                setattr(args, key, value)
        return all_params


