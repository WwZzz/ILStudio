import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, Dataset, DistributedSampler
from typing import Optional, Callable, Any, Dict, Union
import transformers
from transformers import Trainer
import math
import warnings


class _WrappedIterableDataset(IterableDataset):
    """Wrapper for IterableDataset to apply sample-level transformations"""
    
    def __init__(self, dataset: IterableDataset, transform_fn: Callable):
        self.dataset = dataset
        self.transform_fn = transform_fn
        
        # Copy attributes from original dataset if they exist
        if hasattr(dataset, 'dataset_dir'):
            self.dataset_dir = dataset.dataset_dir
    
    def __iter__(self):
        """Apply transform to each sample from the underlying dataset"""
        for sample in self.dataset:
            yield self.transform_fn(sample)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset"""
        return getattr(self.dataset, name)


class _WrappedDataset(Dataset):
    """Wrapper for Dataset to apply sample-level transformations"""
    
    def __init__(self, dataset: Dataset, transform_fn: Callable):
        self.dataset = dataset
        self.transform_fn = transform_fn
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform_fn(sample)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset"""
        return getattr(self.dataset, name)


class _DistributedIterableDataset(IterableDataset):
    """Wrapper for IterableDataset to handle distributed training
    
    This class ensures that each process gets a different subset of the data
    and handles batching internally for IterableDataset.
    """
    
    def __init__(
        self, 
        dataset: IterableDataset, 
        batch_size: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size or (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = rank or (dist.get_rank() if dist.is_initialized() else 0)
        self.seed = seed
        
        # Copy attributes from original dataset if they exist
        if hasattr(dataset, 'dataset_dir'):
            self.dataset_dir = dataset.dataset_dir
    
    def __iter__(self):
        """Iterate with distributed logic and internal batching"""
        worker_info = torch.utils.data.get_worker_info()
        
        # Calculate skip and take for distributed training
        # Each rank should process different data
        iterator = iter(self.dataset)
        
        # Skip samples for other ranks
        samples_to_skip = self.rank
        for _ in range(samples_to_skip):
            try:
                next(iterator)
            except StopIteration:
                return
        
        # Collect samples into batches
        batch = []
        sample_count = 0
        
        for sample in iterator:
            # Take every world_size-th sample for this rank
            if sample_count % self.world_size == 0:
                batch.append(sample)
                
                # Yield batch when it's full
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            
            sample_count += 1
        
        # Yield remaining samples if any
        if batch:
            yield batch
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying dataset"""
        return getattr(self.dataset, name)


def create_iter_loader(
    dataset: IterableDataset, 
    batch_size: int, 
    num_parallels: int,
    collate_fn: Optional[Callable] = None,
    sample_transform_fn: Optional[Callable] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    enable_distributed_batching: bool = True,
    seed: int = 42,
    *args, 
    **kwargs
) -> DataLoader:
    """Create data loader from IterableDataset that can be used by transformers.Trainer
    
    Args:
        dataset: IterableDataset instance
        batch_size: Batch size per device
        num_parallels: Number of parallel workers
        collate_fn: Batch-level processing function
        sample_transform_fn: Sample-level processing function
        drop_last: Whether to drop the last incomplete batch
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker
        persistent_workers: Whether to keep workers alive between epochs
        enable_distributed_batching: Whether to enable internal distributed batching
        seed: Random seed for distributed sampling
    
    Returns:
        DataLoader configured for IterableDataset
    """
    
    # Apply sample-level transform if provided
    if sample_transform_fn is not None:
        dataset = _WrappedIterableDataset(dataset, sample_transform_fn)
    
    # Handle distributed training with internal batching for IterableDataset
    if (enable_distributed_batching and 
        dist.is_available() and 
        dist.is_initialized() and 
        dist.get_world_size() > 1):
        
        # Wrap dataset to handle distributed logic and internal batching
        dataset = _DistributedIterableDataset(
            dataset, 
            batch_size=batch_size,
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            seed=seed
        )
        
        # When using internal batching, DataLoader batch_size should be 1
        # because the dataset already returns batches
        dataloader_batch_size = 1
        
        # Use identity collate function since batching is done internally
        if collate_fn is None:
            collate_fn = lambda x: x[0]  # Extract the batch from the list
    else:
        dataloader_batch_size = batch_size
    
    # For IterableDataset, we don't use sampler (it's None by default)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,  # IterableDataset handles shuffling internally
        sampler=None,   # No sampler for IterableDataset
        batch_sampler=None,
        num_workers=num_parallels,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        **kwargs
    )
    
    return data_loader


def create_map_loader(
    dataset: Dataset, 
    batch_size: int, 
    num_parallels: int,
    collate_fn: Optional[Callable] = None,
    sample_transform_fn: Optional[Callable] = None,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    seed: int = 42,
    *args, 
    **kwargs
) -> DataLoader:
    """Create data loader from torch.utils.data.Dataset that can be used by transformers.Trainer. 
    Distributed Sampler must be used for this loader
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size per device
        num_parallels: Number of parallel workers
        collate_fn: Batch-level processing function
        sample_transform_fn: Sample-level processing function
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of samples loaded in advance by each worker
        persistent_workers: Whether to keep workers alive between epochs
        seed: Random seed for distributed sampler
    
    Returns:
        DataLoader configured for Dataset with DistributedSampler
    """
    
    # Apply sample-level transform if provided
    if sample_transform_fn is not None:
        dataset = _WrappedDataset(dataset, sample_transform_fn)
    
    # Create distributed sampler for multi-GPU training
    sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        # When using DistributedSampler, shuffle should be False in DataLoader
        shuffle = False
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=None,
        num_workers=num_parallels,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        **kwargs
    )
    
    return data_loader


class BaseTrainer(Trainer):
    """Base trainer that extends transformers.Trainer to work with custom data loaders
    
    This trainer can be initialized from data loaders created by create_iter_loader or create_map_loader.
    It handles both IterableDataset and regular Dataset with proper distributed training support.
    """

    def __init__(
        self, 
        model,
        args: transformers.TrainingArguments,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], transformers.PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[transformers.EvalPrediction], Dict]] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        *trainer_args, 
        **trainer_kwargs
    ):
        """
        Initialize BaseTrainer with custom data loaders
        
        Args:
            model: The model to train
            args: Training arguments
            train_loader: Training data loader (can be None if train_dataset is provided)
            eval_loader: Evaluation data loader (can be None if eval_dataset is provided)
            tokenizer: Tokenizer for the model
            model_init: Function to initialize model
            compute_metrics: Function to compute metrics
            callbacks: List of callbacks
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits
        """
        
        # Store custom data loaders
        self._custom_train_loader = train_loader
        self._custom_eval_loader = eval_loader
        
        # Extract datasets from loaders if provided
        train_dataset = train_loader.dataset if train_loader is not None else None
        eval_dataset = eval_loader.dataset if eval_loader is not None else None
        data_collator = None
        
        # Get collate function from loader if available
        if train_loader is not None and hasattr(train_loader, 'collate_fn'):
            data_collator = train_loader.collate_fn
        elif eval_loader is not None and hasattr(eval_loader, 'collate_fn'):
            data_collator = eval_loader.collate_fn
            
        # Initialize parent Trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **trainer_kwargs
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """Override to use custom train loader if provided"""
        if self._custom_train_loader is not None:
            return self._custom_train_loader
        return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """Override to use custom eval loader if provided"""
        if self._custom_eval_loader is not None:
            return self._custom_eval_loader
        return super().get_eval_dataloader(eval_dataset)
    
    def _set_signature_columns_if_needed(self):
        """Override to handle custom datasets that might not have standard columns"""
        try:
            super()._set_signature_columns_if_needed()
        except Exception as e:
            # If signature detection fails, use a more permissive approach
            warnings.warn(f"Could not detect model signature columns: {e}. Using permissive mode.")
            self._signature_columns = None
    
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """Override to handle custom data formats"""
        try:
            return super()._prepare_input(data)
        except Exception:
            # If standard preparation fails, return data as-is
            # This allows for custom data formats that don't follow HF conventions
            if isinstance(data, (dict, list, tuple)):
                return data
            return data