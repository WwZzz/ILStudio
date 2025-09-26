import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Any, List, Union
import numpy as np


class PI0Trainer(Trainer):
    """Custom trainer for PI0 policy."""
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for PI0 training."""
        # Forward pass
        loss, loss_dict = model(inputs)
        
        if return_outputs:
            return loss, loss_dict
        return loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prediction step for evaluation."""
        model.eval()
        
        with torch.no_grad():
            # Forward pass
            loss, loss_dict = model(inputs)
            
            # Get predictions
            predictions = model.predict_action_chunk(inputs)
            
            # Convert to numpy for metrics computation
            if predictions is not None:
                predictions = predictions.detach().cpu().numpy()
            
            if prediction_loss_only:
                return loss.detach().cpu(), None, None
            
            # For evaluation, we might want to return targets
            targets = inputs.get('action', None)
            if targets is not None:
                targets = targets.detach().cpu().numpy()
            
            return loss.detach().cpu(), predictions, targets


def create_pi0_trainer(
    model_name_or_path: str,
    training_args: TrainingArguments,
    train_dataset=None,
    eval_dataset=None,
    data_collator=None,
    compute_metrics=None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PI0Trainer:
    """Create a PI0 trainer."""
    
    # Load model
    from .modeling_pi0 import PI0FlowMatching, PI0FlowMatchingConfig
    
    if config is None:
        config = PI0FlowMatchingConfig()
    
    model = PI0FlowMatching(config)
    
    # Load pretrained weights if available
    if model_name_or_path and model_name_or_path != "scratch":
        try:
            # Try to load from checkpoint
            checkpoint = torch.load(model_name_or_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {model_name_or_path}: {e}")
            print("Training from scratch...")
    
    # Create trainer
    trainer = PI0Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **kwargs
    )
    
    return trainer


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    if predictions is None or labels is None:
        return {}
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Compute basic metrics
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }
