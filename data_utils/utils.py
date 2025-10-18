import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
import json
import torchvision.transforms as transforms
# import IPython  # Removed to avoid unnecessary dependency
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

# e = IPython.embed  # Removed to avoid unnecessary dependency

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
    Append normalization meta information to json file
    """
    # If the file does not exist, create it with an empty placeholder
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'state':{}, 'action':{}}, f)

    # Open with r+ mode to read first then write
    with open(file_path, 'r+', encoding='utf-8') as f:
        # Read existing content
        try:
            old = json.load(f)
        except json.JSONDecodeError:
            # File is empty or corrupted, reinitialize
            old = {'state':{}, 'action':{}}

        # Move pointer to the beginning of file
        f.seek(0)
        # Append new dict
        old['state'].update(data.get('state', {}))
        old['action'].update(data.get('action', {}))
        old['kwargs'] = data.get('kwargs', {})
        # Write back
        json.dump(old, f, ensure_ascii=False, indent=2)
        # Truncate excess content (when new content is shorter than old content)
        f.truncate()

def load_normalizer_from_meta(dataset_dir:str='', norm_meta=None, src_dir=''):
    assert norm_meta is not None, "norm_meta cannot be None "
    if isinstance(norm_meta, str):
        with open(norm_meta, 'r') as f:
            norm_meta = json.load(f)
    kwargs = norm_meta.get('kwargs', {})
    if dataset_dir=='': 
        # when dataset_dir is not specified, using the first dataset dir in normalize.json
        dataset_dir = list(norm_meta['state'].keys())[0]
        warnings.warn(f"dataset_dir was not specified. using {dataset_dir} as the default value.")
    if src_dir=='': src_dir = dataset_dir
    dname = BaseNormalizer.meta2name(dataset_dir=dataset_dir, ctrl_space=kwargs.get('ctrl_space', 'ee'), ctrl_type=kwargs.get('ctrl_type', 'delta'))
    state_normalizer = NORMTYPE2CLASS[norm_meta['state'][dataset_dir]](src_dir, dataset_name=dname, **kwargs)
    action_normalizer = NORMTYPE2CLASS[norm_meta['action'][dataset_dir]](src_dir, dataset_name=dname, **kwargs)
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
        Dataset instance
    """
    # Get dataset class
    class_path = dataset_config.get('class', dataset_config.get('dataset_class', 'EpisodicDataset'))
    dataset_class = _import_class_from_path(class_path)
    
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
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset {class_path} with args {final_args}: {e}")


def load_data(args, task_config, save_norm=True):
    """Load datasets with flexible configuration support
    
    Supports both legacy format and new flexible format:
    
    Legacy format (backward compatible):
    ```yaml
    dataset_dir: ['path1', 'path2']
    dataset_class: 'EpisodicDataset'
    camera_names: ['primary']
    # ... other params
    ```
    
    New flexible format:
    ```yaml
    datasets:
      - name: "main_dataset"
        class: "data_utils.datasets.EpisodicDataset"
        args:
          dataset_dir: ['path1']
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
    
    # Check if using new flexible format
    if 'datasets' in task_config:
        return _load_data_flexible_format(args, task_config, save_norm)
    else:
        return _load_data_legacy_format(args, task_config, save_norm)


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
    
    for dataset in datasets:
        dataset_dir = dataset.get_dataset_dir()
        action_normalizers[dataset_dir] = action_normalizer_class(dataset)
        state_normalizers[dataset_dir] = state_normalizer_class(dataset)
    
    # Save normalization metadata
    if save_norm:
        # Get ctrl_space and ctrl_type from first dataset or task config
        ctrl_space = task_config.get('ctrl_space', 'ee')
        ctrl_type = task_config.get('ctrl_type', 'delta')
        
        # Try to get from first dataset if not in task config
        if len(datasets) > 0 and hasattr(datasets[0], 'ctrl_space'):
            ctrl_space = getattr(datasets[0], 'ctrl_space', ctrl_space)
            ctrl_type = getattr(datasets[0], 'ctrl_type', ctrl_type)
        
        norm_meta = {
            'state': {k: str(v) for k, v in state_normalizers.items()}, 
            'action': {k: str(v) for k, v in action_normalizers.items()}, 
            'kwargs': {'ctrl_space': ctrl_space, 'ctrl_type': ctrl_type}
        }
        save_norm_meta_to_json(os.path.join(args.output_dir, 'normalize.json'), norm_meta)
        
        for dataset_dir, normalizer in state_normalizers.items():
            try:
                normalizer.save_stats_to_(args.output_dir)
            except Exception as e:
                print(f"Failed to save normalizer stats of {dataset_dir} because {e}")
    
    # Wrap datasets with normalizers
    from data_utils.dataset_wrappers import wrap_dataset_with_normalizers
    wrapped_datasets = []
    for dataset in datasets:
        dataset_name = dataset.get_dataset_dir() if hasattr(dataset, 'get_dataset_dir') else None
        wrapped_dataset = wrap_dataset_with_normalizers(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_name
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


def _load_data_legacy_format(args, task_config, save_norm=True):
    """Load data using the legacy configuration format (backward compatibility)"""
    
    dataset_dir_l = task_config['dataset_dir']
    camera_names = task_config.get('camera_names', [])
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 1.0)
    ctrl_space = task_config.get('ctrl_space', 'ee')
    ctrl_type = task_config.get('ctrl_type', 'delta')
    data_class = task_config.get('dataset_class', 'EpisodicDataset')
    
    action_normtype = getattr(args, 'action_normalize', task_config.get('action_normalize', 'zscore'))
    state_normtype = getattr(args, 'state_normalize', task_config.get('state_normalize', 'zscore'))
    
    if isinstance(dataset_dir_l, str):
        dataset_dir_l = [dataset_dir_l]
    
    # Import dataset class
    if data_class == 'EpisodicDataset':
        from data_utils.datasets import EpisodicDataset
        data_class = EpisodicDataset
    else:
        data_class = _import_class_from_path(data_class)
    
    # Create datasets
    rank = dist.get_rank() if is_distributed() else 0
    if rank == 0:
        datasets = [
            data_class(
                [dataset_dir], 
                camera_names, 
                data_args=args, 
                chunk_size=getattr(args, 'chunk_size', 16), 
                ctrl_space=ctrl_space, 
                ctrl_type=ctrl_type
            ) 
            for dataset_dir in dataset_dir_l
        ]
    
    if is_distributed():
        dist.barrier()
        
    if rank != 0:
        datasets = [
            data_class(
                [dataset_dir], 
                camera_names, 
                data_args=args, 
                chunk_size=getattr(args, 'chunk_size', 16), 
                ctrl_space=ctrl_space, 
                ctrl_type=ctrl_type
            ) 
            for dataset_dir in dataset_dir_l
        ]
    
    # Get normalizer classes
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    
    # Compute dataset statistics
    action_normalizers = {dataset.get_dataset_dir(): action_normalizer_class(dataset) for dataset in datasets}
    state_normalizers = {dataset.get_dataset_dir(): state_normalizer_class(dataset) for dataset in datasets}
    
    if save_norm:
        norm_meta = {
            'state': {k: str(v) for k, v in state_normalizers.items()}, 
            'action': {k: str(v) for k, v in action_normalizers.items()}, 
            'kwargs': {'ctrl_space': ctrl_space, 'ctrl_type': ctrl_type}
        }
        save_norm_meta_to_json(os.path.join(args.output_dir, 'normalize.json'), norm_meta)
        
        for dataset_dir, normalizer in state_normalizers.items():
            try:
                normalizer.save_stats_to_(args.output_dir)
            except Exception as e:
                print(f"Failed to save normalizer stats of {dataset_dir} because {e}")
    
    # Wrap datasets with normalizers
    from data_utils.dataset_wrappers import wrap_dataset_with_normalizers
    wrapped_datasets = []
    for dataset in datasets:
        dataset_name = dataset.get_dataset_dir() if hasattr(dataset, 'get_dataset_dir') else None
        wrapped_dataset = wrap_dataset_with_normalizers(
            dataset=dataset,
            action_normalizers=action_normalizers,
            state_normalizers=state_normalizers,
            dataset_name=dataset_name
        )
        wrapped_datasets.append(wrapped_dataset)
        
    train_dataset = ConcatDataset(wrapped_datasets)
    x = train_dataset[0]  # test __getitem__
    val_dataset = None

    return train_dataset, val_dataset

def load_normalizers(args):
    try:
        # load normalizers
        policy_normalize_file = os.path.join(os.path.dirname(args.model_name_or_path), 'normalize.json')
        if not os.path.exists(policy_normalize_file):
            policy_normalize_file = os.path.join(args.model_name_or_path, 'normalize.json')
            if not os.path.exists(policy_normalize_file):
                raise FileNotFoundError("No normalize.json found")
        with open(policy_normalize_file, 'r') as f:
            norm_meta = json.load(f)
        normalizers = load_normalizer_from_meta(args.dataset_dir, norm_meta, os.path.dirname(policy_normalize_file))
        kwargs = norm_meta.get('kwargs', {'ctrl_type':'delta', 'ctrl_space':'ee'})
        return normalizers, kwargs['ctrl_space'], kwargs['ctrl_type'] 
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