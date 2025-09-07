import os
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Any, List, Union
import warnings
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

IGNORE_INDEX = -100


class SimpleOpenVLAConfig(PretrainedConfig):
    """
    Simplified configuration class for OpenVLA Policy.
    """
    def __init__(
        self,
        # Training mode parameters
        training_mode="lora",  # "lora" or "full"
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=False,
        # Model parameters
        max_length=2048,
        # Task parameters
        state_dim=14,
        action_dim=14,
        camera_names=["primary"],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Training mode
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        
        # Model parameters
        self.max_length = max_length
        
        # Task parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.camera_names = camera_names if camera_names is not None else []


class SimpleOpenVLAPolicy(PreTrainedModel):
    """
    Simplified OpenVLA Policy model for robot action prediction.
    """
    config_class = SimpleOpenVLAConfig
    
    def __init__(self, config: SimpleOpenVLAConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize model components (will be loaded from pretrained)
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    def load_pretrained_components(self, model_name_or_path: str):
        """Load pretrained model components."""
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Load model with optional quantization
        quantization_config = None
        if self.config.use_quantization:
            assert self.config.training_mode == "lora", "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid SDPA issues
        )
        
        # Initialize tokenizer
        self.tokenizer = self.processor.tokenizer
        
        # Apply LoRA if needed
        if self.config.training_mode == "lora":
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=min(self.config.lora_alpha, 16),
                lora_dropout=self.config.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.model = get_peft_model(self.model, lora_config)
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of the OpenVLA model.
        """
        if self.model is None:
            raise ValueError("Model components not loaded. Call load_pretrained_components first.")
            
        # Remove num_items_in_batch if present (from trainer)
        kwargs.pop('num_items_in_batch', None)
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def predict_action(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        instruction: str,
        **kwargs
    ) -> np.ndarray:
        """
        Predict action from image and instruction.
        """
        if self.model is None:
            raise ValueError("Model components not loaded. Call load_pretrained_components first.")
            
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            if image.dtype != torch.uint8:
                image = (image * 255).byte()
            image = Image.fromarray(image.numpy())
        
        # Build simple prompt
        prompt = f"Human: What action should the robot take to {instruction}?\nAssistant:"
        
        # Tokenize
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action from response (simplified - just return zeros for now)
        # In practice, you'd need to implement proper action parsing
        try:
            # This is a placeholder - you'd need to implement proper action extraction
            action = np.zeros(self.config.action_dim, dtype=np.float32)
            return action
        except Exception as e:
            warnings.warn(f"Failed to decode action from response: {response}. Error: {e}")
            return np.zeros(self.config.action_dim, dtype=np.float32)
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.config.action_dim
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.config.state_dim
    
    def get_camera_names(self) -> List[str]:
        """Get camera names."""
        return self.config.camera_names


def load_model(args):
    """Load OpenVLA model components."""
    config = SimpleOpenVLAConfig(
        training_mode=getattr(args, 'training_mode', 'lora'),
        lora_r=getattr(args, 'lora_r', 16),
        lora_alpha=getattr(args, 'lora_alpha', 32),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        use_quantization=getattr(args, 'use_quantization', False),
        max_length=getattr(args, 'max_length', 2048),
        state_dim=getattr(args, 'state_dim', 14),
        action_dim=getattr(args, 'action_dim', 14),
        camera_names=getattr(args, 'camera_names', ['primary']),
    )
    
    model = SimpleOpenVLAPolicy(config)
    model.load_pretrained_components(args.model_name_or_path)
    
    return {
        'model': model,
        'tokenizer': model.tokenizer,
        'processor': model.processor
    }


def get_data_collator(args, model_components):
    """Get data collator for OpenVLA."""
    tokenizer = model_components['tokenizer']
    
    class SimpleCollator:
        def __init__(self, tokenizer, dtype=torch.bfloat16):
            self.tokenizer = tokenizer
            self.dtype = dtype
        
        def __call__(self, features):
            # Extract pixel_values, input_ids, and labels
            pixel_values = torch.stack([f['pixel_values'] for f in features])
            input_ids = torch.stack([f['input_ids'] for f in features])
            labels = torch.stack([f['labels'] for f in features])
            
            # Convert pixel_values to the correct dtype
            pixel_values = pixel_values.to(self.dtype)
            
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels
            }
    
    return SimpleCollator(tokenizer)


class SimpleOpenVLAProcessor:
    """Simplified data processor for OpenVLA training."""
    
    def __init__(self, tokenizer, image_transform):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
    
    def __call__(self, sample):
        """Process a single sample."""
        # Handle image format: (num_cameras, C, H, W) -> take first camera
        image_tensor = sample['image'][0]  # Take first camera (primary)
        # Convert from tensor to PIL Image
        image_array = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        image = Image.fromarray(image_array)
        
        # Get action and instruction
        action = sample['action'][0]  # Take first action from chunk
        instruction = sample['raw_lang']
        
        # Build simple conversation with fixed format
        prompt = f"Human: {instruction}\nAssistant: {action.tolist()}"
        
        # Tokenize with fixed length
        inputs = self.tokenizer(
            prompt, 
            add_special_tokens=True, 
            padding="max_length",  # Pad to max_length
            truncation=True, 
            max_length=256,  # Shorter max length
            return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()
        
        # Convert to tensors
        pixel_values = self.image_transform(image)
        
        # Set labels for non-action tokens to IGNORE_INDEX (simplified)
        # In practice, you'd want more sophisticated label masking
        labels[:-len(action)] = IGNORE_INDEX
        
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


def get_data_processor(args, model_components):
    """Get data processor for OpenVLA."""
    tokenizer = model_components['tokenizer']
    image_transform = model_components['processor'].image_processor.apply_transform
    return SimpleOpenVLAProcessor(
        tokenizer=tokenizer,
        image_transform=image_transform
    )
