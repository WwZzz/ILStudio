import os
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Any, List, Union
import warnings
from PIL import Image
import sys
sys.path.append(os.path.dirname(__file__))
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import LlamaTokenizerFast

IGNORE_INDEX = -100

class OpenConfig(PretrainedConfig):
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
        pretrained_weight_path: str="openvla/openvla-7b",
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
        self.pretrained_weight_path = pretrained_weight_path
        

class OpenPolicy(PreTrainedModel):
    """
    Simplified OpenVLA Policy model for robot action prediction.
    """
    config_class = OpenConfig
    
    def __init__(self, config: OpenConfig):
        super().__init__(config)
        self.config = config
        # Initialize model components (will be loaded from pretrained)
        self.processor =  AutoProcessor.from_pretrained(config.pretrained_weight_path, trust_remote_code=True)
        # Initialize tokenizer
        self.tokenizer = self.processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        # Initialize model
        quantization_config = None
        if self.config.use_quantization:
            assert self.config.training_mode == "lora", "Quantized training only supported for LoRA fine-tuning!"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_quant_type="nf4"
            )
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.pretrained_weight_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Use eager attention to avoid SDPA issues
        )
        
    
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
        output = self.model(
            pixel_values=pixel_values.to(torch.bfloat16),
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        action_logits = output.logits[:, self.model.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = labels[:, 1:].to(action_preds.device)
        mask = action_gt > self.action_tokenizer.action_token_begin_idx
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()
        continuous_actions_pred = torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()))
        continuous_actions_gt = torch.tensor(self.action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
        return {'loss': output.loss, 'action_l1_loss': action_l1_loss, 'action_acc': action_accuracy}
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def select_action(
        self,
        obs,
        **kwargs
    ) -> np.ndarray:
        """
        Predict action from image and instruction.
        """
        images = [img[0] for img in obs['image']]
        # Convert image to PIL if needed
        if isinstance(images[0], np.ndarray):
            if images[0].dtype != np.uint8:
                images = [(image * 255).astype(np.uint8) for image in images]
            if len(images[0].shape) == 3 and images[0].shape[0] == 3:  # CHW format
                images = [image.transpose(1, 2, 0) for image in images]
            images = [Image.fromarray(image) for image in images]
        elif isinstance(images[0], torch.Tensor):
            if images[0].dim() == 3 and images[0].shape[0] == 3:  # CHW format
                images[0] = [image.permute(1, 2, 0) for image in images]
            if images[0].dtype != torch.uint8:
                images[0] = [(image * 255).byte() for image in images]
            images = [Image.fromarray(image.numpy()) for image in images]
        instructions = obs['raw_lang']
        # Build simple prompt
        prompts = [f"Human: What action should the robot take to {ins}?\nAssistant:" for ins in instructions]
        # Tokenize
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt"
        )
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device,dtype=torch.bfloat16) if k=='pixel_values' else v.to(device) for k, v in inputs.items()}
        bs = len(prompts)
        if not torch.all(inputs['input_ids'][:, -1] == 29871):
            inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.Tensor([[29871] for _ in range(bs)]).long().to(device)), dim=1)
            inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.Tensor([[1] for _ in range(bs)]).long().to(device)), dim=1)
        inputs = [{k: v[i:i+1,:] if isinstance(v, torch.Tensor) else v[i] for k, v in inputs.items()} for i in range(bs)]
        # Run VLA inference
        generated_ids = [self.model.generate(**inp, max_new_tokens=self.config.action_dim,) for inp in inputs]

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = [gids[0, -self.config.action_dim:].cpu().numpy() for gids in generated_ids]
        normalized_actions = [self.action_tokenizer.decode_token_ids_to_actions(pids) for pids in predicted_action_token_ids]   
        action = np.stack(normalized_actions)
        return action[:,np.newaxis, :]

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
