import numpy as np
import torch
import os
import fnmatch
import json
import warnings
import importlib
from torch.utils.data import DataLoader, ConcatDataset
from .normalize import MinMaxNormalizer, PercentileNormalizer, ZScoreNormalizer, Identity
import torch.distributed as dist

# Import torchdata for mixed dataset support
try:
    from torchdata.datapipes.iter import IterableWrapper, Cycler, ShardingFilter, Shuffler, Batcher, Prefetcher
    TORCHDATA_AVAILABLE = True
except ImportError:
    TORCHDATA_AVAILABLE = False
    warnings.warn("torchdata not available. Multi-dataset mixing will not be supported.")

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
    """
    Create DataLoader from single dataset or multiple datasets.
    
    For multiple datasets (list), uses torchdata to create mixed iterable pipeline with:
    - Cycle: Prevents pipeline exhaustion
    - ShardingFilter: Distributed training support
    - Shuffler: Randomization
    - Batcher: Batching (if needed based on collator)
    - Prefetcher: Performance optimization
    
    Args:
        train_dataset: Single dataset or list of datasets
        val_dataset: Optional validation dataset
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
    
    Returns:
        (train_loader, eval_loader)
    """
    if isinstance(train_dataset, list):
        # Multiple datasets - use torchdata for mixing
        if not TORCHDATA_AVAILABLE:
            raise RuntimeError("torchdata is required for multi-dataset mixing. Install with: pip install torchdata")
        
        print(f"Using mixed dataset pipeline with {len(train_dataset)} datasets")
        train_loader = _create_mixed_dataloader(train_dataset, processor, collator, args)
        
        # Handle validation dataset
        eval_loader = None
        if val_dataset is not None:
            if isinstance(val_dataset, list):
                eval_loader = _create_mixed_dataloader(val_dataset, processor, collator, args, is_training=False)
            else:
                eval_loader = _create_single_dataloader(val_dataset, processor, collator, args, is_training=False)
        
        return train_loader, eval_loader
    
    else:
        # Single dataset - use existing logic
        return _create_single_dataloader(train_dataset, processor, collator, args, is_training=True)


def _create_single_dataloader(dataset, processor, collator, args, is_training=True):
    """Create DataLoader for a single dataset (map-style or iterable)"""
    # Identify the type of the dataset: iter or map
    is_iter_dataset = hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))
    
    if not is_iter_dataset:
        # Map-style dataset
        if hasattr(dataset, '__len__'):
            print(f"{'Train' if is_training else 'Validation'} dataset size: {len(dataset)}")
        
        # Create DataLoader with WrappedDataset
        wrapped_data = WrappedDataset(dataset, processor)
        
        if is_training and is_distributed():
            from torch.utils.data.distributed import DistributedSampler
            print(f"Using DistributedSampler for distributed training")
            sampler = DistributedSampler(wrapped_data) 
        else:
            sampler = None
        
        loader = DataLoader(
            wrapped_data,
            batch_size=args.per_device_train_batch_size,
            shuffle=(sampler is None and is_training),
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
            drop_last=is_training,
            pin_memory=args.dataloader_pin_memory,
        )
        
        if is_training:
            return loader, None
        else:
            return loader
    
    else:
        # Iterable dataset
        print(f"Using {'training' if is_training else 'validation'} iterable dataset")
        
        # Wrap iterable dataset with processor
        wrapped_data = WrappedIterableDataset(dataset, processor)
        
        # For iterable datasets, we cannot use DistributedSampler
        if is_training and is_distributed():
            print(f"Warning: Iterable datasets should handle distributed training internally")
            print(f"Make sure your dataset splits data across workers properly")
        
        # Create DataLoader for iterable dataset
        loader = DataLoader(
            wrapped_data,
            batch_size=args.per_device_train_batch_size,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
            drop_last=is_training,
            pin_memory=args.dataloader_pin_memory,
        )
        
        if is_training:
            return loader, None
        else:
            return loader


def _create_mixed_dataloader(datasets, processor, collator, args, is_training=True):
    """
    Create DataLoader for multiple datasets using torchdata pipeline.
    
    Pipeline structure:
    1. Wrap/Convert each dataset to iterable
    2. Apply processor to each dataset
    3. Wrap as IterableWrapper (torchdata pipeline)
    4. Apply Cycle to prevent exhaustion
    5. Apply ShardingFilter for distributed training
    6. Apply Shuffler for randomization
    7. Apply Batcher (if collator expects unbatched data)
    8. Apply Prefetcher for performance
    9. Create DataLoader
    
    Args:
        datasets: List of datasets (can be map-style or iterable)
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
        is_training: Whether this is for training (affects shuffle, drop_last)
    
    Returns:
        DataLoader with mixed pipeline
    """
    print(f"Building mixed dataset pipeline:")
    
    # Step 1: Convert all datasets to iterable with processor applied
    iterable_datasets = []
    for i, dataset in enumerate(datasets):
        is_iter = hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))
        
        if is_iter:
            # Already iterable
            print(f"  Dataset {i}: Iterable dataset")
            wrapped = WrappedIterableDataset(dataset, processor)
        else:
            # Map-style: convert to iterable
            print(f"  Dataset {i}: Map-style dataset (size={len(dataset)}) -> converting to iterable")
            # First apply processor via WrappedDataset
            wrapped_map = WrappedDataset(dataset, processor)
            # Then convert to iterable
            wrapped = MapToIterableDataset(wrapped_map, shuffle=is_training, seed=args.seed if hasattr(args, 'seed') else None)
        
        iterable_datasets.append(wrapped)
    
    # Step 2: Create torchdata pipelines for each dataset
    print(f"Creating torchdata pipelines:")
    pipelines = []
    for i, dataset in enumerate(iterable_datasets):
        # Wrap as IterableWrapper
        pipe = IterableWrapper(dataset)
        
        # Apply Cycle to prevent exhaustion
        pipe = pipe.cycle()
        print(f"  Pipeline {i}: IterableWrapper -> Cycle")
        
        pipelines.append(pipe)
    
    # Step 3: Multiplex (mix) all pipelines
    # Use round-robin sampling from each pipeline
    from torchdata.datapipes.iter import Multiplexer
    print(f"Multiplexing {len(pipelines)} pipelines")
    mixed_pipe = Multiplexer(*pipelines)
    
    # Step 4: Apply ShardingFilter for distributed training
    if is_distributed():
        print(f"Applying ShardingFilter for distributed training")
        mixed_pipe = mixed_pipe.sharding_filter()
    
    # Step 5: Apply Shuffler for randomization (if training)
    if is_training:
        buffer_size = getattr(args, 'shuffle_buffer_size', 10000)
        print(f"Applying Shuffler with buffer_size={buffer_size}")
        mixed_pipe = mixed_pipe.shuffle(buffer_size=buffer_size)
    
    # Step 6: Batching is handled by DataLoader's batch_size parameter
    # We don't use Batcher here because collator might need custom logic
    
    # Step 7: Apply Prefetcher for performance
    prefetch_size = getattr(args, 'prefetch_size', 10)
    print(f"Applying Prefetcher with buffer_size={prefetch_size}")
    mixed_pipe = mixed_pipe.prefetch(buffer_size=prefetch_size)
    
    # Step 8: Create DataLoader
    print(f"Creating DataLoader with batch_size={args.per_device_train_batch_size}")
    loader = DataLoader(
        mixed_pipe,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collator,
        drop_last=is_training,
        pin_memory=args.dataloader_pin_memory,
    )
    
    return loader
    

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

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class WrappedDataset(torch.utils.data.Dataset):
    """Wrapper for map-style datasets"""
    def __init__(self, dataset, processor=None):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample if self.processor is None else self.processor(sample)


class WrappedIterableDataset(torch.utils.data.IterableDataset):
    """Wrapper for iterable datasets with processor support"""
    def __init__(self, dataset, processor=None):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
    
    def __iter__(self):
        for sample in self.dataset:
            if self.processor is not None:
                sample = self.processor(sample)
            yield sample


class MapToIterableDataset(torch.utils.data.IterableDataset):
    """Convert a map-style dataset to an iterable dataset with optional shuffling"""
    def __init__(self, dataset, shuffle=True, seed=None):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Use worker info for different shuffling per worker
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # In multi-worker setting
                worker_id = worker_info.id
                seed = self.seed if self.seed is not None else 0
                rng = np.random.RandomState(seed + worker_id)
            else:
                # Single worker
                rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        
        for idx in indices:
            yield self.dataset[idx]


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
    
    # # Handle legacy parameters for backward compatibility (only if not in constructor_args)
    # legacy_params = {}
    # if 'dataset_dir' not in constructor_args and 'dataset_path_list' not in constructor_args:
    #     if 'dataset_dir' in dataset_config or 'path' in dataset_config:
    #         legacy_params['dataset_dir'] = dataset_config.get('dataset_dir', dataset_config.get('path'))
    
    # if 'camera_names' not in constructor_args:
    #     legacy_params['camera_names'] = dataset_config.get('camera_names', [])
    
    # if 'chunk_size' not in constructor_args:
    #     legacy_params['chunk_size'] = dataset_config.get('chunk_size', getattr(args, 'chunk_size', 16))
    
    # if 'ctrl_space' not in constructor_args:
    #     legacy_params['ctrl_space'] = dataset_config.get('ctrl_space', 'ee')
    
    # if 'ctrl_type' not in constructor_args:
    #     legacy_params['ctrl_type'] = dataset_config.get('ctrl_type', 'delta')
    
    # Merge legacy params with constructor args (constructor args take priority)
    final_args = {}
    # final_args.update(legacy_params)
    final_args.update(constructor_args)
    
    # # Add data_args if the dataset expects it
    # if 'data_args' not in final_args:
    #     final_args['data_args'] = args
    
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
    train_data = wrapped_datasets[0] if len(wrapped_datasets) == 1 else wrapped_datasets
    return {'train': train_data, 'eval': None}

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