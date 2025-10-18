import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
import json
import torchvision.transforms as transforms
import copy
import gc
import warnings
import importlib
from time import time
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.transforms.functional import to_pil_image, to_tensor
from .normalize import BaseNormalizer, MinMaxNormalizer, PercentileNormalizer, ZScoreNormalizer, Identity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

# Normalize Class
NORMTYPE2CLASS = {
    'minmax': MinMaxNormalizer,
    'percentile': PercentileNormalizer, 
    'zscore': ZScoreNormalizer,
    'identity': Identity,
}

def get_dataloader(train_dataset, val_dataset=None, processor=None, collator=None, args=None):
    # Identify the type of the dataset: iter or map
    is_iter_dataset = hasattr(train_dataset, '__iter__') and not hasattr(train_dataset, '__getitem__')
    if not is_iter_dataset:
        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"Validation dataset size: {len(val_dataset)}")
        # Create DataLoader
        wrapped_train_data = WrappedDataset(train_dataset, processor)
        if is_distributed():
            from torch.utils.data.distributed import DistributedSampler
            print(f"Using DistributedSampler for distributed training")
            sampler = DistributedSampler(wrapped_train_data) 
        else:
            sampler = None
        train_loader = DataLoader(
            wrapped_train_data,
            batch_size=args.per_device_train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
            drop_last=True,
            pin_memory=args.dataloader_pin_memory,
        )
        eval_loader = None
        return train_loader, eval_loader
    else:
        raise NotImplementedError("Iterable dataset is not supported yet.")
    

def find_all_hdf5(dataset_dir):
    """
    Find all HDF5 files in the dataset directory.
    Note: This function is deprecated. Use dataset class internal method instead.
    """
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        if 'pointcloud' in root: continue
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            hdf5_files.append(os.path.join(root, filename))
    return hdf5_files


# def smooth_base_action(base_action):
#     return np.stack([
#         np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
#     ], axis=-1).astype(np.float32)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor=None):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample if self.processor is None else self.processor(sample)


def save_norm_meta_to_json(file_path: str, data: dict):
    """
    Save normalization meta information to json file
    
    Saves complete metadata including datasets (with per-dataset ctrl_space/ctrl_type), state, and action.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_normalizer_from_meta(norm_meta, src_dir='', dataset_id=None):
    """Load normalizers from metadata
    
    Uses dataset_id as keys and includes per-dataset ctrl_space/ctrl_type.
    
    Args:
        norm_meta: Metadata dictionary
        src_dir: Source directory where normalize.json and stats files are located.
                 If empty or stats not found, will fallback to cache directory.
        dataset_id: Specific dataset_id to load. If None, loads the first dataset from metadata.
                    Note: This is read from metadata, not from args.dataset_dir.
    
    Returns:
        Dictionary with 'state' and 'action' normalizers
    """
    datasets_info = norm_meta.get('datasets', [])
    
    if not datasets_info:
        raise ValueError("No datasets found in metadata")
    
    # Determine which dataset to load
    if dataset_id is None:
        # Use first dataset as default (typical for single-dataset training)
        dataset_meta = datasets_info[0]
        dataset_id = dataset_meta['dataset_id']
        if len(datasets_info) > 1:
            print(f"Multiple datasets found in metadata. Using first: {dataset_id}")
    else:
        # Find the matching dataset by dataset_id
        dataset_meta = None
        for ds_meta in datasets_info:
            if ds_meta['dataset_id'] == dataset_id:
                dataset_meta = ds_meta
                break
        
        if dataset_meta is None:
            raise ValueError(f"Dataset '{dataset_id}' not found in metadata. Available: {[d['dataset_id'] for d in datasets_info]}")
    
    # Get ctrl info from metadata
    ctrl_space = dataset_meta.get('ctrl_space', 'ee')
    ctrl_type = dataset_meta.get('ctrl_type', 'delta')
    
    # Get normalizer types from metadata
    state_norm_type = norm_meta['state'].get(dataset_id, 'Zscore')
    action_norm_type = norm_meta['action'].get(dataset_id, 'Zscore')
    
    # Create normalizers with dataset_id and ctrl info
    kwargs = {'ctrl_space': ctrl_space, 'ctrl_type': ctrl_type}
    
    # Determine load directory: prefer src_dir, fallback to cache
    cache_dir = os.path.join(os.environ.get('ILSTD_CACHE', os.path.expanduser('~/.cache/ilstd')), 'normalize')
    
    # Check if stats exist in src_dir
    if src_dir and os.path.exists(src_dir):
        stats_filename = f"{dataset_id}_stats_{ctrl_space}_{ctrl_type}.pkl"
        src_stats_path = os.path.join(src_dir, stats_filename)
        
        if os.path.exists(src_stats_path):
            # Load from src_dir (checkpoint directory)
            load_dir = src_dir
        else:
            # Fallback to cache
            load_dir = cache_dir
            warnings.warn(f"Stats not found in {src_dir}, using cache directory: {cache_dir}")
    else:
        # Use cache directory
        load_dir = cache_dir
    
    # Create normalizers
    state_normalizer = NORMTYPE2CLASS[state_norm_type](
        load_dir, dataset_name=dataset_id, **kwargs
    )
    action_normalizer = NORMTYPE2CLASS[action_norm_type](
        load_dir, dataset_name=dataset_id, **kwargs
    )
    
    return {'state': state_normalizer, 'action': action_normalizer}
    
def _import_class_from_path(class_path: str):
    """Dynamically import a class from a module path
    
    Args:
        class_path: Full path to class, e.g., 'data_utils.datasets.EpisodicDataset'
                   or 'data_utils.datasets.rlds_wrapper.WrappedTFDSDataset'
    
    Returns:
        The imported class
    """
    if '.' not in class_path:
        # If no module path, assume it's in data_utils.datasets
        class_path = f'data_utils.datasets.{class_path}'
    
    module_path, class_name = class_path.rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {class_path}: {e}")


def _create_dataset_from_config(dataset_config: dict, args):
    """Create a dataset instance from configuration
    
    Args:
        dataset_config: Individual dataset configuration
        args: Training arguments
    
    Returns:
        Dataset instance with added 'name' and 'dataset_id' attributes
    """
    # Get dataset class
    class_path = dataset_config.get('class', dataset_config.get('dataset_class', 'EpisodicDataset'))
    dataset_class = _import_class_from_path(class_path)
    
    # Extract dataset name from config (required for identification)
    dataset_name = dataset_config.get('name')
    if not dataset_name:
        raise ValueError(f"Dataset configuration must include a 'name' field: {dataset_config}")
    
    # Extract constructor arguments
    constructor_args = dataset_config.get('args', {})
    
    # Handle legacy parameters for backward compatibility (only if not in constructor_args)
    legacy_params = {}
    if 'dataset_dir' not in constructor_args and 'dataset_path_list' not in constructor_args:
        if 'dataset_dir' in dataset_config or 'path' in dataset_config:
            legacy_params['dataset_dir'] = dataset_config.get('dataset_dir', dataset_config.get('path'))
    
    if 'camera_names' not in constructor_args:
        legacy_params['camera_names'] = dataset_config.get('camera_names', [])
    
    if 'chunk_size' not in constructor_args:
        legacy_params['chunk_size'] = dataset_config.get('chunk_size', getattr(args, 'chunk_size', 16))
    
    if 'ctrl_space' not in constructor_args:
        legacy_params['ctrl_space'] = dataset_config.get('ctrl_space', 'ee')
    
    if 'ctrl_type' not in constructor_args:
        legacy_params['ctrl_type'] = dataset_config.get('ctrl_type', 'delta')
    
    # Merge legacy params with constructor args (constructor args take priority)
    final_args = {}
    final_args.update(legacy_params)
    final_args.update(constructor_args)
    
    # Add data_args if the dataset expects it
    if 'data_args' not in final_args:
        final_args['data_args'] = args
    
    # No automatic parameter conversion - config should match dataset class signature exactly
    
    # Create dataset instance
    try:
        dataset = dataset_class(**final_args)
        
        # Add name and dataset_id attributes for identification
        # These are used by normalizers and other components to identify datasets
        dataset.name = dataset_name
        dataset.dataset_id = dataset_name  # Alias for clarity
        
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset {class_path} with args {final_args}: {e}")


def load_data(args, task_config, save_norm=True):
    """Load datasets with flexible configuration support
    
    Required format:
    ```yaml
    datasets:
      - name: "main_dataset"
        class: "data_utils.datasets.EpisodicDataset"
        args:
          dataset_path_list: ['path1']
          camera_names: ['primary']
          chunk_size: 64
          ctrl_space: 'ee'
          ctrl_type: 'delta'
      - name: "auxiliary_dataset"  
        class: "data_utils.datasets.rlds_wrapper.WrappedTFDSDataset"
        args:
          dataset_path_list: ['path2']
          camera_names: ['primary']
          # ... custom args for this specific dataset
    ```
    """
    
    # Ensure new flexible format is used
    if 'datasets' not in task_config:
        raise ValueError(
            "Task config must use the new flexible format with 'datasets' key. "
            "Old format with 'dataset_dir' is no longer supported. "
            "Please update your task config to use the datasets format."
        )
    
    return _load_data_flexible_format(args, task_config, save_norm)


def _load_data_flexible_format(args, task_config, save_norm=True):
    """Load data using the new flexible configuration format"""
    
    datasets_config = task_config['datasets']
    
    # Get normalization types
    action_normtype = getattr(args, 'action_normalize', task_config.get('action_normalize', 'zscore'))
    state_normtype = getattr(args, 'state_normalize', task_config.get('state_normalize', 'zscore'))
    
    # Create datasets
    rank = dist.get_rank() if is_distributed() else 0
    datasets = []
    
    if rank == 0:
        for dataset_config in datasets_config:
            dataset = _create_dataset_from_config(dataset_config, args)
            datasets.append(dataset)
    
    if is_distributed():
        dist.barrier()
        
    if rank != 0:
        for dataset_config in datasets_config:
            dataset = _create_dataset_from_config(dataset_config, args)
            datasets.append(dataset)
    
    # Compute normalizers
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    
    action_normalizers = {}
    state_normalizers = {}
    
    # Use dataset.dataset_id as the key for normalizers instead of dataset_dir
    for dataset in datasets:
        dataset_id = dataset.dataset_id  # Use the dataset_id attribute added in _create_dataset_from_config
        action_normalizers[dataset_id] = action_normalizer_class(dataset, dataset_name=dataset_id)
        state_normalizers[dataset_id] = state_normalizer_class(dataset, dataset_name=dataset_id)
    
    # Save normalization metadata
    if save_norm:
        # Build complete metadata for each dataset
        datasets_meta = []
        for dataset in datasets:
            dataset_meta = {
                'dataset_id': dataset.dataset_id,
                'ctrl_space': getattr(dataset, 'ctrl_space', 'ee'),
                'ctrl_type': getattr(dataset, 'ctrl_type', 'delta'),
            }
            datasets_meta.append(dataset_meta)
        
        # Metadata format that stores complete information for each dataset
        norm_meta = {
            'version': '2.0',  # Format version
            'datasets': datasets_meta,  # List of dataset metadata
            'state': {k: str(v) for k, v in state_normalizers.items()}, 
            'action': {k: str(v) for k, v in action_normalizers.items()},
        }
        save_norm_meta_to_json(os.path.join(args.output_dir, 'normalize.json'), norm_meta)
        
        # Save normalizer stats to output_dir (for training) using dataset_id as key
        for dataset_id, normalizer in state_normalizers.items():
            try:
                normalizer.save_stats_to_(args.output_dir)
            except Exception as e:
                print(f"Failed to save normalizer stats of {dataset_id} because {e}")
    
    # Wrap datasets with normalizers
    from data_utils.dataset_wrappers import wrap_dataset_with_normalizers
    wrapped_datasets = []
    for dataset in datasets:
        # Use dataset.dataset_id as the identifier instead of dataset_dir
        dataset_id = dataset.dataset_id
        wrapped_dataset = wrap_dataset_with_normalizers(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_id
        )
        wrapped_datasets.append(wrapped_dataset)
    
    # Create combined dataset
    if len(wrapped_datasets) == 1:
        train_dataset = wrapped_datasets[0]
    else:
        train_dataset = ConcatDataset(wrapped_datasets)
    
    # Test dataset
    x = train_dataset[0]  # test __getitem__
    val_dataset = None
    
    return train_dataset, val_dataset

def load_normalizers(args):
    """Load normalizers from saved metadata
    
    Loads normalizers using dataset_id as key with per-dataset ctrl_space/ctrl_type.
    
    Returns:
        tuple: (normalizers_dict, ctrl_space, ctrl_type) or (normalizers_dict, datasets_info)
               For new format, returns list of dataset info dicts
    """
    try:
        # load normalizers
        policy_normalize_file = os.path.join(os.path.dirname(args.model_name_or_path), 'normalize.json')
        if not os.path.exists(policy_normalize_file):
            policy_normalize_file = os.path.join(args.model_name_or_path, 'normalize.json')
            if not os.path.exists(policy_normalize_file):
                raise FileNotFoundError("No normalize.json found")
        with open(policy_normalize_file, 'r') as f:
            norm_meta = json.load(f)
        
        # Load normalizer from metadata (uses first dataset by default)
        # dataset_id is determined from metadata, not from args
        normalizers = load_normalizer_from_meta(
            norm_meta, 
            src_dir=os.path.dirname(policy_normalize_file),
            dataset_id=None  # Let it auto-select from metadata
        )
        
        # Get ctrl info from the loaded dataset (first dataset)
        datasets_info = norm_meta.get('datasets', [])
        if datasets_info:
            first_dataset = datasets_info[0]
            ctrl_space = first_dataset.get('ctrl_space', 'ee')
            ctrl_type = first_dataset.get('ctrl_type', 'delta')
        else:
            ctrl_space, ctrl_type = 'ee', 'delta'
        
        return normalizers, ctrl_space, ctrl_type
            
    except Exception as e:
        warnings.warn(f"Failed to load normalizers from {args.model_name_or_path} because {e}")
        identity_normalizer = {'state':Identity(), 'action':Identity()}
        return identity_normalizer, 'ee', 'delta'

def _convert_to_type(value):
    """
    Infers the type of a value based on its format. Supports int, float, and bool.
    """
    if not isinstance(value, str): return value
    # Attempt to infer boolean value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    # Attempt to infer integer type
    if value.isdigit():
        return int(value)
    # Attempt to infer float type
    try:
        return float(value)
    except ValueError:
        pass
    # Otherwise, return the original string
    return value