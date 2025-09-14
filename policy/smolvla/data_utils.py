import torch
import numpy as np
from typing import Dict, Any, List, Union
from PIL import Image
import torchvision.transforms as transforms


class SmolVLAProcessor:
    """Data processor for SmolVLA training and inference."""
    
    def __init__(self, tokenizer, image_transform=None):
        self.tokenizer = tokenizer
        self.image_transform = image_transform or self._default_image_transform()
    
    def _default_image_transform(self):
        """Default image transformation."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample."""
        # Handle image format: (num_cameras, C, H, W) -> take first camera
        if 'image' in sample:
            image_tensor = sample['image'][0]  # Take first camera (primary)
            # Convert from tensor to PIL Image
            if isinstance(image_tensor, torch.Tensor):
                image_array = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                image = Image.fromarray(image_array)
            else:
                image = image_tensor
        else:
            # Create dummy image if no image provided
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get action and instruction
        action = sample.get('action', torch.zeros(14))  # Default action dimension
        if isinstance(action, (list, np.ndarray)):
            action = torch.tensor(action, dtype=torch.float32)
        
        instruction = sample.get('raw_lang', sample.get('task', 'perform task'))
        if isinstance(instruction, (list, np.ndarray)):
            instruction = str(instruction[0]) if len(instruction) > 0 else 'perform task'
        
        # Process image
        pixel_values = self.image_transform(image)
        
        # Tokenize instruction
        tokenized = self.tokenizer(
            instruction,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=48,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'action': action,
            'task': instruction
        }


def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Data collator for SmolVLA training."""
    batch = {}
    
    # Stack tensors
    for key in ['pixel_values', 'input_ids', 'attention_mask', 'action']:
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
    
    # Handle task (text) separately
    if 'task' in features[0]:
        batch['task'] = [f['task'] for f in features]
    
    return batch


def get_data_processor(args, model_components):
    """Get data processor for SmolVLA."""
    tokenizer = model_components.get('tokenizer')
    if tokenizer is None:
        raise ValueError("Tokenizer not found in model components")
    
    return SmolVLAProcessor(tokenizer=tokenizer)


def get_data_collator(args, model_components):
    """Get data collator for SmolVLA."""
    return data_collator
