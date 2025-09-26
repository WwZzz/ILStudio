# Data utilities for SmolVLA adapted to IL-Studio
import torch
import numpy as np
from typing import Dict, Any, List, Union


class SmolVLAProcessor:
    """Data processor that converts IL-Studio samples to SmolVLA format.
    
    Converts from IL-Studio format:
    - 'image': (K, C, H, W) numpy/tensor in [0,255] or [0,1]
    - 'state'/'state_joint'/'state_ee': (state_dim,) numpy/tensor
    - 'action': (action_dim,) numpy/tensor
    - 'raw_lang'/'task': str instruction
    
    To SmolVLA format:
    - 'observation.images.<camera_name>': (C, H, W) tensor in [0,1]
    - 'observation.state': (state_dim,) tensor
    - 'action': (action_dim,) tensor
    - 'task': str instruction
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.camera_names = getattr(config, 'camera_names', ['primary'])
        self.state_dim = getattr(config, 'state_dim', 14)
        self.action_dim = getattr(config, 'action_dim', 14)

    def _to_tensor(self, arr, dtype=torch.float32):
        """Convert array to tensor with specified dtype."""
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=dtype)
        return torch.tensor(arr, dtype=dtype)

    def _normalize_image(self, img_tensor):
        """Normalize image to [0,1] range."""
        if img_tensor.dtype in (torch.uint8, torch.int32, torch.int16, torch.int64):
            img_tensor = img_tensor.float()
        if img_tensor.max() > 1.1:  # Assume [0,255] if max > 1
            img_tensor = img_tensor / 255.0
        return img_tensor.clamp(0, 1)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample."""
        output = {}
        
        # Process images
        if 'image' in sample and sample['image'] is not None:
            images = sample['image']
            if isinstance(images, (list, tuple)):
                images = np.array(images)
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            elif isinstance(images, torch.Tensor):
                images = images.float()
            
            # Ensure (K, C, H, W) format
            if images.ndim == 3:  # (C, H, W)
                images = images.unsqueeze(0)  # (1, C, H, W)
            
            k, c, h, w = images.shape
            for i, camera_name in enumerate(self.camera_names):
                if i < k:
                    img = self._normalize_image(images[i])  # (C, H, W) in [0,1]
                    output[f'observation.images.{camera_name}'] = img
                else:
                    # Create dummy image if not enough cameras
                    output[f'observation.images.{camera_name}'] = torch.zeros(c, h, w, dtype=torch.float32)
        else:
            # Create dummy images for all cameras
            for camera_name in self.camera_names:
                output[f'observation.images.{camera_name}'] = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        # Process state
        state = None
        for state_key in ['state', 'state_joint', 'state_ee']:
            if state_key in sample and sample[state_key] is not None:
                state = sample[state_key]
                break
        
        if state is None:
            state = torch.zeros(self.state_dim, dtype=torch.float32)
        else:
            state = self._to_tensor(state, torch.float32)
            if state.ndim == 0:
                state = state.unsqueeze(0)
        
        output['observation.state'] = state
        
        # Process action
        action = sample.get('action')
        if action is None:
            action = torch.zeros(self.action_dim, dtype=torch.float32)
        else:
            action = self._to_tensor(action, torch.float32)
            if action.ndim == 0:
                action = action.unsqueeze(0)
        
        output['action'] = action
        
        # Process task/instruction
        task = sample.get('raw_lang', sample.get('task', 'perform the task'))
        if isinstance(task, (list, tuple)):
            task = task[0] if len(task) > 0 else 'perform the task'
        
        output['task'] = str(task)
        
        return output


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch collator for SmolVLA training."""
    if not features:
        return {}
    
    batch = {}
    all_keys = set()
    for feature in features:
        all_keys.update(feature.keys())
    
    for key in all_keys:
        values = [f[key] for f in features if key in f]
        if not values:
            continue
            
        if key == 'task':
            # Keep task as list of strings
            batch[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                batch[key] = torch.stack(values)
            except Exception:
                # Handle variable shapes with padding
                batch[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
        elif isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values
    
    return batch


def get_data_processor(args, model_components):
    """Get data processor for SmolVLA following IL-Studio policy rules."""
    tokenizer = model_components.get('tokenizer')
    if tokenizer is None:
        raise ValueError("Tokenizer not found in model components")
    
    config = model_components.get('config')
    if config is None:
        raise ValueError("Config not found in model components")
    
    return SmolVLAProcessor(tokenizer=tokenizer, config=config)


def get_data_collator(args, model_components):
    """Get data collator for SmolVLA following IL-Studio policy rules."""
    return data_collator

