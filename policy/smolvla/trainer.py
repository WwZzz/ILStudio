# SmolVLA trainer adapted to IL-Studio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Any, List, Union
import numpy as np


class SmolVLATrainer(Trainer):
    """Custom trainer for SmolVLA policy adapted to IL-Studio."""
    
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
    
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None):
        """Compute loss for SmolVLA training."""
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

