import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
import warnings
from PIL import Image

from .simple_openvla import SimpleOpenVLAPolicy as OpenVLAPolicy, SimpleOpenVLAConfig as OpenVLAPolicyConfig, SimpleOpenVLAProcessor as OpenVLAProcessor


class OpenVLATrainer(Trainer):
    """
    Custom trainer for OpenVLA policy.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_tokenizer = None
        self.tokenizer = None
        
    def set_tokenizers(self, action_tokenizer, tokenizer):
        """Set tokenizers for evaluation."""
        self.action_tokenizer = action_tokenizer
        self.tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for OpenVLA model.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Temporarily disable logging
        old_log_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(100)  # Disable logging
        
        try:
            # Run evaluation
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            
            # Calculate metrics
            metrics = output.metrics.copy()
            
            # Add custom metrics if needed
            if hasattr(output, 'predictions') and output.predictions is not None:
                # Calculate action prediction accuracy or other metrics
                pass
                
        finally:
            # Restore logging level
            self.logger.setLevel(old_log_level)
            
        return metrics
    
    def predict_action(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        instruction: str,
        **kwargs
    ) -> np.ndarray:
        """
        Predict action from image and instruction.
        """
        if self.action_tokenizer is None or self.tokenizer is None:
            raise ValueError("Tokenizers not set. Call set_tokenizers first.")
            
        return self.model.predict_action(image, instruction, **kwargs)
    
    def predict_actions_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        instructions: List[str],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Predict actions for a batch of images and instructions.
        """
        actions = []
        for image, instruction in zip(images, instructions):
            action = self.predict_action(image, instruction, **kwargs)
            actions.append(action)
        return actions
    
    def save_model(self, output_dir: str, **kwargs):
        """
        Save the model and tokenizers.
        """
        # Save the main model
        super().save_model(output_dir, **kwargs)
        
        # Save tokenizers if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        # Save action tokenizer (if it has a save method)
        if self.action_tokenizer is not None and hasattr(self.action_tokenizer, 'save_pretrained'):
            self.action_tokenizer.save_pretrained(output_dir)
        
        # Save model config
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[OpenVLAPolicyConfig] = None,
        **kwargs
    ):
        """
        Load a pretrained OpenVLA model and create trainer.
        """
        if config is None:
            config = OpenVLAPolicyConfig()
            
        model = OpenVLAPolicy(config)
        model.load_pretrained_components(model_name_or_path)
        
        # Create trainer with the loaded model
        trainer = cls(model=model, **kwargs)
        
        # Set tokenizers
        trainer.set_tokenizers(
            action_tokenizer=model.action_tokenizer,
            tokenizer=model.tokenizer
        )
        
        return trainer


def create_openvla_trainer(
    model_name_or_path: str,
    training_args: TrainingArguments,
    train_dataset=None,
    eval_dataset=None,
    data_collator=None,
    compute_metrics=None,
    config: Optional[OpenVLAPolicyConfig] = None,
    **kwargs
) -> OpenVLATrainer:
    """
    Create an OpenVLA trainer with the given parameters.
    """
    if config is None:
        config = OpenVLAPolicyConfig()
    
    # Load model
    model = OpenVLAPolicy(config)
    model.load_pretrained_components(model_name_or_path)
    
    # Create trainer
    trainer = OpenVLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **kwargs
    )
    
    # Set tokenizers
    trainer.set_tokenizers(
        action_tokenizer=model.action_tokenizer,
        tokenizer=model.tokenizer
    )
    
    return trainer
