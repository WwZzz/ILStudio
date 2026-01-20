"""
DataLoader for replay buffer.

This module provides a DataLoader-like wrapper for replay buffer with
on-demand normalization and transforms.
"""

from typing import Dict, Any, Optional, List, Callable
import gc
import torch
import numpy as np

from .sampling import sample_processed


class ReplayBufferDataLoader:
    """
    DataLoader-like wrapper for replay buffer with on-demand normalization and transforms.
    
    NEW SAMPLING STRATEGY:
    - Replay buffer stores single-step actions (no duplication)
    - Action chunks are built dynamically during sampling
    - Episode boundaries are respected for padding
    
    Workflow (when buffer stores raw data):
    1. Sample episodes and start positions
    2. Build action chunks with is_pad based on episode boundaries
    3. Apply normalization (state/action normalizers)
    4. Apply transforms (data augmentation, etc.)
    5. Apply data_processor (policy-specific processing)
    6. Collate using data_collator
    
    This provides maximum flexibility for:
    - Runtime data augmentation (can change transforms without reloading data)
    - Experimenting with different normalization strategies
    - Different chunk sizes at sampling time
    
    Memory Management:
    - Periodically runs gc.collect() to avoid memory buildup
    - Set gc_interval to control how often GC runs
    """
    
    def __init__(
        self,
        replay_buffer,
        batch_size: int,
        num_batches_per_epoch: int = 1000,
        data_processor=None,
        data_collator=None,
        device: str = "cuda:0",
        gc_interval: int = 5,
        apply_normalization: bool = True,
        apply_transforms: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            replay_buffer: ILReplayBuffer instance (stores single-step actions)
            batch_size: Number of samples per batch
            num_batches_per_epoch: Number of batches to generate per epoch
            data_processor: Policy-specific processor function
            data_collator: Collator function for batching
            device: Target device for batches
            gc_interval: Garbage collection frequency (0 to disable)
            apply_normalization: Whether to apply normalizers during sampling
            apply_transforms: Whether to apply transforms during sampling
            chunk_size: Action chunk size (default: replay_buffer.chunk_size)
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.data_processor = data_processor
        self.data_collator = data_collator
        self.device = device
        self.gc_interval = gc_interval
        self.apply_normalization = apply_normalization
        self.apply_transforms = apply_transforms
        self.chunk_size = chunk_size if chunk_size is not None else replay_buffer.chunk_size
        
    def __len__(self):
        return self.num_batches_per_epoch
    
    def __iter__(self):
        for i in range(self.num_batches_per_epoch):
            yield self.sample_batch()
            # Periodic garbage collection to prevent memory buildup
            if self.gc_interval > 0 and (i + 1) % self.gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def sample_batch(self):
        """Sample and process a batch with full pipeline.
        
        NEW: Samples episodes, builds action chunks dynamically with proper padding.
        Memory-optimized: Cleans up intermediate objects promptly.
        """
        # Get processed samples (episode sampling → action chunk building → normalization → transforms → processor)
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            self.data_processor,
            device='cpu',  # Keep on CPU, collator/move will handle device
            apply_normalization=self.apply_normalization,
            apply_transforms=self.apply_transforms,
            chunk_size=self.chunk_size,
        )
        
        # Collate
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        # Clear samples list immediately after collation (no longer needed)
        del samples
        
        # Move to device and return
        result = self._to_device(batch, self.device)
        
        # Clear batch dict to free memory (tensors are already moved to device)
        del batch
        
        return result
    
    def sample_batch_raw(self):
        """Sample a batch of raw data without any processing."""
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            data_processor=None,
            device='cpu',
            apply_normalization=False,
            apply_transforms=False,
        )
        
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        return self._to_device(batch, self.device)
    
    def sample_batch_normalized_only(self):
        """Sample a batch with only normalization (no transforms or processor)."""
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            data_processor=None,
            device='cpu',
            apply_normalization=True,
            apply_transforms=False,
        )
        
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        return self._to_device(batch, self.device)
    
    def set_transforms(self, transforms):
        """
        Update transforms at runtime.
        
        This allows changing data augmentation without reloading the buffer.
        
        Args:
            transforms: New transform pipeline (list of callables)
        """
        self.replay_buffer.transforms = transforms
    
    def set_normalizers(self, action_normalizer=None, state_normalizer=None):
        """
        Update normalizers at runtime.
        
        Args:
            action_normalizer: New action normalizer
            state_normalizer: New state normalizer
        """
        if action_normalizer is not None:
            self.replay_buffer.action_normalizer = action_normalizer
        if state_normalizer is not None:
            self.replay_buffer.state_normalizer = state_normalizer
    
    def _default_collate(self, samples):
        """Stack tensors, keep lists for non-tensors."""
        if not samples:
            return {}
        batch = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples if key in s]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch
    
    def _to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch
    
    def get_data_info(self):
        """Get information about the data pipeline configuration."""
        return {
            'batch_size': self.batch_size,
            'num_batches_per_epoch': self.num_batches_per_epoch,
            'buffer_size': self.replay_buffer.size,
            'buffer_capacity': self.replay_buffer.capacity,
            'store_raw': self.replay_buffer.store_raw,
            'apply_normalization': self.apply_normalization,
            'apply_transforms': self.apply_transforms,
            'has_action_normalizer': self.replay_buffer.action_normalizer is not None,
            'has_state_normalizer': self.replay_buffer.state_normalizer is not None,
            'has_transforms': self.replay_buffer.transforms is not None,
            'has_processor': self.data_processor is not None,
            'has_collator': self.data_collator is not None,
        }

