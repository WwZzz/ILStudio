import numpy as np
import torch
import os
import fnmatch
import json
import importlib
from loguru import logger
import torch
import torch.distributed as dist
# import dlimp as dl # Added this import
from PIL import Image
from torch.utils.data import IterableDataset
from .normalize import NORMTYPE2CLASS, save_norm_meta_to_json

class RatioSplittingIterableDataset(IterableDataset):
    """
    Wraps an iterable dataset to split it into train and eval sets by ratio on the fly.
    This is a deterministic split based on the sample index.
    """
    def __init__(self, dataset, eval_ratio, mode='train', seed=0):
        super().__init__()
        self.dataset = dataset
        if not (0 < eval_ratio < 1):
            raise ValueError("eval_ratio must be between 0 and 1.")
        self.eval_ratio = eval_ratio
        self.mode = mode
        self.seed = seed

    def __iter__(self):
        # Use a generator for deterministic "random" decisions based on index
        g = torch.Generator()
        g.manual_seed(self.seed)

        for i, sample in enumerate(self.dataset):
            # Generate a random number for each sample
            rand_val = torch.rand(1, generator=g).item()
            
            # Decide whether to yield the sample based on the mode
            is_eval_sample = rand_val < self.eval_ratio
            
            if self.mode == 'train' and not is_eval_sample:
                yield sample
            elif self.mode == 'eval' and is_eval_sample:
                yield sample

def save_example_data(train_data, output_dir):
    """
    Save example data from the first dataset for debugging purposes.
    Saves raw (unnormalized) data to match testing phase format:
    - Images: original resolution, saved as PNG
    - State: raw unnormalized values, saved as state_raw.csv
    - Action: raw unnormalized values, saved as action_raw.csv
    
    Args:
        train_data: The dataset object or list of datasets (can be map-style or iterable)
        output_dir: Directory to save the example data
    """
    try:
        # Create directory for examples
        examples_dir = os.path.join(output_dir, 'example_data')
        
        # Check if example data already exists
        if os.path.exists(examples_dir):
            # Check if any example files exist
            existing_files = os.listdir(examples_dir)
            if len(existing_files) > 0:
                logger.info(f"Example data already exists in {examples_dir}, skipping save.")
                return
        
        os.makedirs(examples_dir, exist_ok=True)
        
        # Handle list of datasets or single dataset
        if isinstance(train_data, list):
            if len(train_data) == 0:
                logger.warning("Empty dataset list provided")
                return
            dataset = train_data[0]  # Use first dataset
            logger.info(f"Saving example from first dataset (list of {len(train_data)} datasets)")
        else:
            dataset = train_data
            logger.info("Saving example from single dataset")
        
        # Get one sample from the dataset (raw, before any processing)
        # Check if dataset is map-style (has __getitem__) or iterable
        sample = None
        if hasattr(dataset, '__getitem__'):
            # Map-style dataset
            try:
                sample = dataset[0]
            except Exception as e:
                logger.warning(f"Could not get sample from map-style dataset: {e}")
                return
        else:
            # Iterable dataset
            try:
                sample = next(iter(dataset))
            except Exception as e:
                logger.warning(f"Could not get sample from iterable dataset: {e}")
                return
        
        if sample is None:
            logger.warning("Could not retrieve sample from dataset")
            return
        
        # Save raw language instruction
        if 'raw_lang' in sample and sample['raw_lang']:
            lang_file = os.path.join(examples_dir, 'raw_lang.txt')
            with open(lang_file, 'w', encoding='utf-8') as f:
                f.write(str(sample['raw_lang']))
            logger.info(f"Saved language instruction to: {lang_file}")
        
        # Save images - save each camera view separately, preserving original resolution
        # Match testing phase format: save as camera_{i}.png without resizing
        if 'image' in sample and sample['image'] is not None:
            image_data = sample['image']
            
            # Convert tensor to numpy if needed
            if isinstance(image_data, torch.Tensor):
                image_data = image_data.cpu().numpy()
            
            logger.debug(f"Training example - Image shape: {image_data.shape}, dtype: {image_data.dtype}")
            
            # Handle different image formats
            # Expected format: (num_cameras, C, H, W) or (C, H, W)
            if len(image_data.shape) == 4:  # Multiple cameras: (num_cameras, C, H, W)
                num_cameras = image_data.shape[0]
                
                for cam_idx in range(num_cameras):
                    img = image_data[cam_idx]  # (C, H, W)
                    # Convert from (C, H, W) to (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                    # Normalize to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Save individual camera image
                    image_file = os.path.join(examples_dir, f'camera_{cam_idx}.png')
                    pil_image = Image.fromarray(img)
                    pil_image.save(image_file)
                    logger.info(f"Saved camera {cam_idx} image (shape: {img.shape}) to: {image_file}")
                
            elif len(image_data.shape) == 3:  # Single camera: (C, H, W)
                img = image_data
                # Convert from (C, H, W) to (H, W, C)
                img = np.transpose(img, (1, 2, 0))
                # Normalize to 0-255 if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # Save single camera image
                image_file = os.path.join(examples_dir, 'camera_0.png')
                pil_image = Image.fromarray(img)
                pil_image.save(image_file)
                logger.info(f"Saved single camera image (shape: {img.shape}) to: {image_file}")
            else:
                logger.warning(f"Unexpected image shape: {image_data.shape}")
        
        # Save state as separate CSV (raw, unnormalized)
        # Match testing phase format: state_raw.csv
        if 'state' in sample and sample['state'] is not None:
            state_data = sample['state']
            if isinstance(state_data, torch.Tensor):
                state_data = state_data.cpu().numpy()
            
            # Ensure state_data is at least 2D for np.savetxt
            if state_data.ndim == 1:
                state_data = state_data.reshape(1, -1)
            elif state_data.ndim > 2:
                # Flatten to 2D if needed
                state_data = state_data.reshape(1, -1)
            
            state_file = os.path.join(examples_dir, 'state_raw.csv')
            header = ','.join([f'state_{i}' for i in range(state_data.shape[1])])
            np.savetxt(state_file, state_data, delimiter=',', header=header, comments='')
            logger.info(f"Saved raw state (unnormalized) to: {state_file}")
        
        # Save action as separate CSV (raw, unnormalized)
        # Match testing phase format: action_raw.csv
        if 'action' in sample and sample['action'] is not None:
            action_data = sample['action']
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.cpu().numpy()
            
            # Action might be (chunk_size, action_dim) or (action_dim,)
            if action_data.ndim == 1:
                action_data = action_data.reshape(1, -1)
            elif action_data.ndim > 2:
                # Flatten higher dimensions
                original_shape = action_data.shape
                action_data = action_data.reshape(-1, action_data.shape[-1])
                logger.debug(f"Reshaped action from {original_shape} to {action_data.shape}")
            
            action_file = os.path.join(examples_dir, 'action_raw.csv')
            header = ','.join([f'action_{i}' for i in range(action_data.shape[1])])
            np.savetxt(action_file, action_data, delimiter=',', header=header, comments='')
            logger.info(f"Saved raw action (unnormalized) to: {action_file}")
        
        # Save reasoning as JSON if not empty
        if 'reasoning' in sample and sample['reasoning']:
            reasoning = sample['reasoning']
            # Check if reasoning is not empty
            if reasoning and (not isinstance(reasoning, str) or reasoning.strip()):
                reasoning_file = os.path.join(examples_dir, 'reasoning.json')
                with open(reasoning_file, 'w', encoding='utf-8') as f:
                    if isinstance(reasoning, dict):
                        json.dump(reasoning, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump({'reasoning': str(reasoning)}, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved reasoning to: {reasoning_file}")
        
        # Save metadata info file to match testing phase format
        info_file = os.path.join(examples_dir, 'info.txt')
        with open(info_file, 'w') as f:
            f.write("=== Training Example Data Info ===\n\n")
            f.write("This example is saved from the raw training dataset (before data_processor).\n")
            f.write("Data is saved in unnormalized form to match testing phase format.\n\n")
            
            # Sample info
            f.write("Sample keys:\n")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    shape = value.shape if hasattr(value, 'shape') else 'N/A'
                    dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
                    f.write(f"  {key}: shape={shape}, dtype={dtype}\n")
                elif value is not None:
                    f.write(f"  {key}: {type(value).__name__}\n")
            
            f.write("\nFiles saved:\n")
            f.write("  - camera_{i}.png: raw images from dataset (unnormalized, original resolution)\n")
            f.write("  - state_raw.csv: raw state values (unnormalized)\n")
            f.write("  - action_raw.csv: raw action values (unnormalized)\n")
            f.write("  - raw_lang.txt: language instruction (if available)\n")
            f.write("  - reasoning.json: reasoning data (if available)\n")
            f.write("  - info.txt: this file\n\n")
            f.write("Note: These raw values can be directly compared with testing phase examples.\n")
        
        logger.info(f"Saved example info to: {info_file}")
        logger.info("Successfully saved example data from first dataset (raw, unnormalized)")
        
    except Exception as e:
        logger.error(f"Error saving example data: {e}")
        import traceback
        traceback.print_exc()

def safe_decode(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif isinstance(value, (int, np.integer)):
        return str(int(value))
    else:
        return str(value)

def ensure_uint8_image(image_array):
    """
    Ensure image array is uint8 with values in [0, 255] range.
    Handles conversion from normalized float images (0-1) or other formats.
    
    Args:
        image_array: numpy array or torch tensor
        
    Returns:
        Image array in uint8 format with values [0, 255]
    """
    # Handle torch tensors directly without numpy conversion
    if isinstance(image_array, torch.Tensor):
        if image_array.dtype == torch.uint8:
            return image_array
        
        if image_array.dtype in [torch.float32, torch.float64, torch.float16]:
            # Normalized float image (0-1)
            if image_array.max() <= 1.0 and image_array.min() >= 0.0:
                return (image_array * 255).clamp(0, 255).to(torch.uint8)
            # Float image already in [0, 255] range
            else:
                return image_array.clamp(0, 255).to(torch.uint8)
        else:
            # Other dtypes (int32, int64, etc.)
            return image_array.clamp(0, 255).to(torch.uint8)
    
    # Handle numpy arrays
    else:
        if image_array.dtype == np.uint8:
            return image_array
        
        if image_array.dtype in [np.float32, np.float64, np.float16]:
            # Normalized float image (0-1)
            if image_array.max() <= 1.0 and image_array.min() >= 0.0:
                return (image_array * 255).clip(0, 255).astype(np.uint8)
            # Float image already in [0, 255] range
            else:
                return image_array.clip(0, 255).astype(np.uint8)
        else:
            # Other dtypes (int32, int64, etc.)
            return image_array.clip(0, 255).astype(np.uint8)

def convert_rlds_sample(data):
    # Ensure images are uint8 [0, 255]
    image_data = ensure_uint8_image(data['image'])
    
    data_dict = dict(
        raw_lang = safe_decode(data['raw_lang']),
        image = torch.einsum('k h w c -> k c h w', torch.from_numpy(image_data) if isinstance(image_data, np.ndarray) else image_data),
        state = torch.from_numpy(data['state']).float(),
        action = torch.from_numpy(data['action']).float(),
    )
    if 'is_pad' in data:
        data_dict['is_pad'] = torch.from_numpy(data['is_pad']).bool()
    if 'timestamp' in data:
        data_dict['timestamp'] = data['timestamp']
    if 'episode_id' in data:
        data_dict['episode_id'] = safe_decode(data['episode_id'])
    if 'dataset_id' in data:
        data_dict['dataset_id'] = safe_decode(data['dataset_id'])
    return data_dict

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

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
    """Set all random seeds to ensure reproducibility
    
    Args:
        seed: random seed
    """
    import random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multiple GPUs
    
    # cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # environment variable (some libraries will read)
    os.environ['PYTHONHASHSEED'] = str(seed)

def flatten_list(l):
    return [item for sublist in l for item in sublist]
    
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
    # Get dataset class - support both new 'type' and old 'class' fields
    class_path = dataset_config.get('type') or dataset_config.get('class') or dataset_config.get('dataset_class', 'EpisodicDataset')
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

def _normalize_datasets(datasets, args, task_config, save_norm=True):
    datasets_config = task_config['datasets']
    # Get normalization types
    action_normtype = getattr(args, 'action_normalize', task_config.get('action_normalize', 'zscore'))
    state_normtype = getattr(args, 'state_normalize', task_config.get('state_normalize', 'zscore'))
    # Compute normalizers
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    
    action_normalizers = {}
    state_normalizers = {}
    
    # Use dataset.dataset_id as the key for normalizers instead of dataset_dir
    # Also extract mask information from dataset_config
    for dataset, dataset_config in zip(datasets, datasets_config):
        dataset_id = dataset.dataset_id  # Use the dataset_id attribute added in _create_dataset_from_config
        
        # Extract mask information from dataset config (same level as args)
        action_norm_mask = dataset_config.get('action_norm_mask', None)
        state_norm_mask = dataset_config.get('state_norm_mask', None)
        
        # Log mask configuration for transparency
        if action_norm_mask is not None or state_norm_mask is not None:
            logger.info(f"Creating normalizers with mask configuration for dataset '{dataset_id}':")
            if action_norm_mask is not None:
                logger.info(f"  - action_norm_mask: {action_norm_mask}")
            if state_norm_mask is not None:
                logger.info(f"  - state_norm_mask: {state_norm_mask}")
        
        # Create normalizers with masks
        action_normalizers[dataset_id] = action_normalizer_class(
            dataset, 
            dataset_name=dataset_id, 
            mask=action_norm_mask
        )
        state_normalizers[dataset_id] = state_normalizer_class(
            dataset, 
            dataset_name=dataset_id, 
            mask=state_norm_mask
        )
    
    # Save normalization metadata
    if save_norm:
        # Build complete metadata for each dataset
        datasets_meta = []
        for dataset, dataset_config in zip(datasets, datasets_config):
            # Extract mask info from config (same level as args)
            action_norm_mask = dataset_config.get('action_norm_mask', None)
            state_norm_mask = dataset_config.get('state_norm_mask', None)
            
            dataset_meta = {
                'dataset_id': dataset.dataset_id,
                'ctrl_space': getattr(dataset, 'ctrl_space', 'ee'),
                'ctrl_type': getattr(dataset, 'ctrl_type', 'delta'),
            }
            
            # Add mask information if present
            if action_norm_mask is not None:
                # Convert to list for JSON serialization
                if isinstance(action_norm_mask, np.ndarray):
                    dataset_meta['action_norm_mask'] = action_norm_mask.tolist()
                else:
                    dataset_meta['action_norm_mask'] = action_norm_mask
            
            if state_norm_mask is not None:
                # Convert to list for JSON serialization
                if isinstance(state_norm_mask, np.ndarray):
                    dataset_meta['state_norm_mask'] = state_norm_mask.tolist()
                else:
                    dataset_meta['state_norm_mask'] = state_norm_mask
            
            datasets_meta.append(dataset_meta)
        
        # Metadata format that stores complete information for each dataset
        norm_meta = {
            'version': '2.0',  # Format version
            'datasets': datasets_meta,  # List of dataset metadata
            'state': {k: str(v) for k, v in state_normalizers.items()}, 
            'action': {k: str(v) for k, v in action_normalizers.items()}, 
        }
        
        # Log mask information being saved
        has_mask = any('action_norm_mask' in ds or 'state_norm_mask' in ds for ds in datasets_meta)
        if has_mask:
            logger.info(f"Saving normalizer metadata with mask configurations to: {os.path.join(args.output_dir, 'normalize.json')}")
            for ds_meta in datasets_meta:
                if 'action_norm_mask' in ds_meta or 'state_norm_mask' in ds_meta:
                    logger.info(f"  Dataset '{ds_meta['dataset_id']}':")
                    if 'action_norm_mask' in ds_meta:
                        logger.info(f"    - action_norm_mask: {ds_meta['action_norm_mask']}")
                    if 'state_norm_mask' in ds_meta:
                        logger.info(f"    - state_norm_mask: {ds_meta['state_norm_mask']}")
        
        save_norm_meta_to_json(os.path.join(args.output_dir, 'normalize.json'), norm_meta)
        
        # Save normalizer stats to output_dir (for training) using dataset_id as key
        for dataset_id, normalizer in state_normalizers.items():
            try:
                normalizer.save_stats_to_(args.output_dir)
            except Exception as e:
                logger.warning(f"Failed to save normalizer stats of {dataset_id} because {e}")
    
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
    return wrapped_datasets

def _train_val_split_datasets(datasets, args):
    """Split datasets into train and eval sets"""
        # Data splitting logic based on eval_ratio and dataset types
    eval_ratio = getattr(args, 'eval_ratio', 0.0)
    train_data_splits = []
    eval_data_splits = []

    if eval_ratio > 0:
        logger.info(f"Splitting each dataset with eval_ratio: {eval_ratio}")
        for ds in datasets:
            # --- Handle RLDS Datasets ---
            if hasattr(ds, 'rlds_dataset') and hasattr(ds.rlds_dataset, 'split'):
                train_rlds_ds, eval_rlds_ds = ds.rlds_dataset.split(
                    [1.0 - eval_ratio, eval_ratio], deterministic=True, drop_remainder=False
                )

                # Re-create instances of the original class with the new split datasets
                # This preserves all other configurations of the dataset wrapper
                train_ds_split = ds.__class__.__new__(ds.__class__)
                train_ds_split.__dict__ = ds.__dict__.copy()
                train_ds_split.rlds_dataset = train_rlds_ds
                train_ds_split.dataset = train_rlds_ds.dataset

                eval_ds_split = ds.__class__.__new__(ds.__class__)
                eval_ds_split.__dict__ = ds.__dict__.copy()
                eval_ds_split.rlds_dataset = eval_rlds_ds
                eval_ds_split.dataset = eval_rlds_ds.dataset
                
                train_data_splits.append(train_ds_split)
                eval_data_splits.append(eval_ds_split)
                logger.info(f"Split WrappedRLDSDataset '{getattr(ds, 'name', 'N/A')}' by ratio {eval_ratio}.")

            # --- Handle Map-style Datasets ---
            elif is_map_data(ds):
                num_total = len(ds)
                if num_total == 0:
                    logger.warning(f"Dataset '{getattr(ds.dataset, 'name', 'N/A')}' is empty, skipping split.")
                    continue
                
                num_eval = int(num_total * eval_ratio)
                if num_eval == 0 and num_total > 1:
                    num_eval = 1
                
                num_train = num_total - num_eval
                if num_train <= 0 and num_total > 1:
                    num_train = num_total - 1
                    num_eval = 1
                if num_train > 0 or num_eval > 0:
                    train_split, eval_split = torch.utils.data.random_split(
                        ds, [num_train, num_eval],
                        generator=torch.Generator().manual_seed(getattr(args, 'seed', 0)) if getattr(args, 'seed', None) is not None else None
                    )
                    if len(train_split) > 0: train_data_splits.append(train_split)
                    if len(eval_split) > 0: eval_data_splits.append(eval_split)
                    # logger.info(f"Split map-style dataset '{getattr(ds.dataset, 'name', 'N/A')}': {len(train_split)} train, {len(eval_split)} eval.")
                else:
                    train_data_splits.append(ds)
                    logger.warning(f"Could not split map-style dataset '{getattr(ds.dataset, 'name', 'N/A')}' with {num_total} samples. Added to train set.")

            # --- Handle Generic Iterable Datasets ---
            else:
                logger.info(f"Splitting generic iterable dataset '{getattr(ds, 'name', 'N/A')}' by ratio {eval_ratio}.")
                train_split = RatioSplittingIterableDataset(ds, eval_ratio, mode='train', seed=getattr(args, 'seed', 0))
                eval_split = RatioSplittingIterableDataset(ds, eval_ratio, mode='eval', seed=getattr(args, 'seed', 0))
                train_data_splits.append(train_split)
                eval_data_splits.append(eval_split)

    else:  # eval_ratio is 0 or less
        # Per user request, do not create an eval set if eval_ratio is not specified (or is <= 0).
        # All datasets will be used for training.
        logger.info("eval_ratio <= 0. All datasets will be used for training. No evaluation dataset will be created.")
        train_data_splits.extend(datasets)
        # eval_data_splits remains an empty list, so eval_data will be None.

    # --- Finalize train and eval data ---
    train_data = None
    if len(train_data_splits) == 1:
        train_data = train_data_splits[0]
    elif len(train_data_splits) > 1:
        train_data = train_data_splits # Return as a list for _create_mixed_dataloader

    eval_data = None
    if len(eval_data_splits) == 1:
        eval_data = eval_data_splits[0]
    elif len(eval_data_splits) > 1:
        eval_data = eval_data_splits
    return train_data, eval_data

def _apply_transforms_to_datasets(datasets, args, task_config):
    """Apply transforms to datasets
    Args:
        datasets: List of datasets
        args: Training arguments
        task_config: Task configuration
    Returns:
        List of transformed datasets
    """
    transform_configs = task_config.get('transforms', [])
    if not transform_configs or len(transform_configs) == 0:
        return datasets
    # dynamically import the transform class
    from .transform import TransformPipeline, MapTransformPipeline, IterableTransformPipeline
    transform_pipes = []
    
    for transform_config in transform_configs:
        transform_class = _import_class_from_path(transform_config.get('type'))
        transform = transform_class(**transform_config.get('args', {}))
        transform_pipes.append(transform)
    transform_pipe = TransformPipeline(transform_pipes)
    transformed_datasets = []
    for dataset in datasets:
        if is_map_data(dataset):
            transformed_datasets.append(MapTransformPipeline(dataset, transform_pipe))
        elif is_iter_data(dataset):
            transformed_datasets.append(IterableTransformPipeline(dataset, transform_pipe))
        else:
            raise ValueError(f"Dataset type {type(dataset)} not supported for transformation.")
    return transformed_datasets

def _maybe_assign_weights_to_datasets(datasets, task_config):
    sample_weights = task_config.get('sample_weights', None)
    if sample_weights is None or not isinstance(datasets, list): return datasets
    datasets_config = task_config['datasets']
    if isinstance(sample_weights, dict):
        sample_weights = [sample_weights.get(dcfg['name'], 1.0) for dcfg in datasets_config]
    for wi, dataset_i in zip(sample_weights, datasets):
        dataset_i.__weight__ = wi
    return datasets

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
    datasets_config = task_config['datasets']
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
    # Normalize datasets
    datasets = _normalize_datasets(datasets, args, task_config, save_norm)

    # Apply transforms to datasets
    datasets = _apply_transforms_to_datasets(datasets, args, task_config)

    # Split datasets into train and eval sets
    train_data, eval_data = _train_val_split_datasets(datasets, args)

    # Assigning weights to each dataset
    train_data = _maybe_assign_weights_to_datasets(train_data, task_config)
    
    return {'train': train_data, 'eval': eval_data}

def is_rlds_data(ds):
    import dlimp as dl # Ensure dlimp is imported here as well if needed
    return isinstance(ds, dl.DLataset)

def is_map_data(dataset):
    return hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')

def is_iter_data(dataset):
    return hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))

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