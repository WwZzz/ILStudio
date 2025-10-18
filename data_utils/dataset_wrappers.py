"""
Dataset wrappers for applying normalization without modifying the original dataset classes.

This module provides wrapper classes that apply normalization to dataset outputs,
supporting both map-style and iterable datasets.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, Optional, Any
import copy


class NormalizedMapDataset(Dataset):
    """
    Wrapper for map-style datasets that applies normalization to actions and states.
    
    This wrapper preserves the map-style nature of the dataset while transparently
    applying normalization to the output samples.
    
    Args:
        dataset: The underlying map-style dataset (torch.utils.data.Dataset)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        action_normalizers: Optional[Dict] = None,
        state_normalizers: Optional[Dict] = None,
        dataset_name: Optional[str] = None
    ):
        self.dataset = dataset
        self.action_normalizers = action_normalizers or {}
        self.state_normalizers = state_normalizers or {}
        
        # Determine dataset name for normalizer lookup
        if dataset_name is None:
            # Try to get dataset_dir from the dataset
            if hasattr(dataset, 'dataset_dir'):
                dataset_name = dataset.dataset_dir
            elif hasattr(dataset, 'dataset_path_list') and len(dataset.dataset_path_list) > 0:
                dataset_name = dataset.dataset_path_list[0]
            elif hasattr(dataset, 'get_dataset_dir'):
                dataset_name = dataset.get_dataset_dir()
            else:
                dataset_name = 'default'
        
        self.dataset_name = dataset_name
        
        # Get the specific normalizers for this dataset
        self.action_normalizer = self.action_normalizers.get(dataset_name, None)
        self.state_normalizer = self.state_normalizers.get(dataset_name, None)
    
    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset and apply normalization.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Sample dictionary with normalized actions and states
        """
        # Get the original sample from the underlying dataset
        sample = self.dataset[idx]
        
        # Make a copy to avoid modifying the original
        sample = copy.copy(sample)
        
        # Apply action normalization if available
        if self.action_normalizer is not None and 'action' in sample:
            sample['action'] = self.action_normalizer.normalize(sample['action'])
        
        # Apply state normalization if available
        if self.state_normalizer is not None and 'state' in sample:
            sample['state'] = self.state_normalizer.normalize(sample['state'])
        
        return sample
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataset.
        
        This allows the wrapper to be transparent and expose all methods
        and attributes of the wrapped dataset.
        """
        # Avoid infinite recursion for special attributes
        if name in ['dataset', 'action_normalizers', 'state_normalizers', 
                    'dataset_name', 'action_normalizer', 'state_normalizer']:
            return object.__getattribute__(self, name)
        
        # Forward to the underlying dataset
        return getattr(self.dataset, name)


class NormalizedIterableDataset(IterableDataset):
    """
    Wrapper for iterable datasets that applies normalization to actions and states.
    
    This wrapper preserves the iterable nature of the dataset while transparently
    applying normalization to the output samples.
    
    Args:
        dataset: The underlying iterable dataset (torch.utils.data.IterableDataset)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        action_normalizers: Optional[Dict] = None,
        state_normalizers: Optional[Dict] = None,
        dataset_name: Optional[str] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.action_normalizers = action_normalizers or {}
        self.state_normalizers = state_normalizers or {}
        
        # Determine dataset name for normalizer lookup
        if dataset_name is None:
            # Try to get dataset name from the dataset
            if hasattr(dataset, 'dataset_dir'):
                dataset_name = dataset.dataset_dir
            elif hasattr(dataset, 'dataset_path_list') and len(dataset.dataset_path_list) > 0:
                dataset_name = dataset.dataset_path_list[0]
            elif hasattr(dataset, 'dataset_path'):
                dataset_name = dataset.dataset_path
            else:
                dataset_name = 'default'
        
        self.dataset_name = dataset_name
        
        # Get the specific normalizers for this dataset
        self.action_normalizer = self.action_normalizers.get(dataset_name, None)
        self.state_normalizer = self.state_normalizers.get(dataset_name, None)
    
    def __iter__(self):
        """
        Iterate over the dataset and apply normalization to each sample.
        
        Yields:
            Sample dictionary with normalized actions and states
        """
        for sample in self.dataset:
            # Make a copy to avoid modifying the original
            sample = copy.copy(sample)
            
            # Apply action normalization if available
            if self.action_normalizer is not None and 'action' in sample:
                sample['action'] = self.action_normalizer.normalize(sample['action'])
            
            # Apply state normalization if available
            if self.state_normalizer is not None and 'state' in sample:
                sample['state'] = self.state_normalizer.normalize(sample['state'])
            
            yield sample
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying dataset.
        
        This allows the wrapper to be transparent and expose all methods
        and attributes of the wrapped dataset.
        """
        # Avoid infinite recursion for special attributes
        if name in ['dataset', 'action_normalizers', 'state_normalizers', 
                    'dataset_name', 'action_normalizer', 'state_normalizer']:
            return object.__getattribute__(self, name)
        
        # Forward to the underlying dataset
        return getattr(self.dataset, name)


def wrap_dataset_with_normalizers(
    dataset,
    action_normalizers: Optional[Dict] = None,
    state_normalizers: Optional[Dict] = None,
    dataset_name: Optional[str] = None
):
    """
    Automatically wrap a dataset with the appropriate normalizer wrapper.
    
    This function uses duck typing to detect whether the dataset is map-style 
    or iterable by checking for the presence of __getitem__ and __iter__ methods.
    
    Args:
        dataset: The dataset to wrap (either map-style or iterable)
        action_normalizers: Dictionary mapping dataset names/paths to action normalizers
        state_normalizers: Dictionary mapping dataset names/paths to state normalizers
        dataset_name: Name or path of the dataset (used to lookup the correct normalizer)
    
    Returns:
        Wrapped dataset with normalization applied
    """
    # Check if normalization is needed
    if not action_normalizers and not state_normalizers:
        # No normalizers provided, return original dataset
        return dataset
    
    # Use duck typing to detect dataset type
    # Check for __getitem__ (map-style) vs __iter__ (iterable)
    has_getitem = hasattr(dataset, '__getitem__') and callable(getattr(dataset, '__getitem__'))
    has_iter = hasattr(dataset, '__iter__') and callable(getattr(dataset, '__iter__'))
    has_len = hasattr(dataset, '__len__') and callable(getattr(dataset, '__len__'))
    
    # Map-style datasets have both __getitem__ and __len__
    # Iterable datasets have __iter__ but may not have __getitem__ or __len__
    if has_getitem and has_len:
        # Map-style dataset
        return NormalizedMapDataset(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_name
        )
    elif has_iter:
        # Iterable dataset
        return NormalizedIterableDataset(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_name
        )
    else:
        # Unknown dataset type, return as-is and let it fail later if needed
        import warnings
        warnings.warn(
            f"Dataset type {type(dataset)} does not implement standard dataset interface "
            f"(__getitem__+__len__ for map-style or __iter__ for iterable). "
            f"Returning unwrapped dataset."
        )
        return dataset

