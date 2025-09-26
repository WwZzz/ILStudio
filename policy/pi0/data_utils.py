import torch
import numpy as np
from typing import Dict, Any, List, Union
from PIL import Image
import torchvision.transforms as transforms


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


class PI0Processor:
    """Data processor for PI0 training and inference."""
    
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
    """Data collator for PI0 training."""
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
    """Get data processor for PI0."""
    tokenizer = model_components.get('tokenizer')
    if tokenizer is None:
        raise ValueError("Tokenizer not found in model components")
    
    return PI0Processor(tokenizer=tokenizer)


def get_data_collator(args, model_components):
    """Get data collator for PI0."""
    return data_collator

class DataCollator:
    def __init__(self, config, language_tokenizer, tokenizer_max_length, resize_imgs_with_padding, image_features, empty_cameras):
        self.config = config
        self.empty_cameras = empty_cameras
        self.language_tokenizer = language_tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.image_features = image_features

    def __call__(self, batch):
        return {
            'images': ..., 
            'img_masks': ...,
            'lang_tokens':..., 
            'lang_masks':..., 
            'state':..., 
            'actions':..., 
            'is_pad':..., 
            'noise':None, 
            'time': None,
        }

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.image_features if key in batch]
        missing_img_keys = [key for key in self.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            if self.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.resize_imgs_with_padding, pad_value=0)
            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        return lang_tokens, lang_masks


    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
