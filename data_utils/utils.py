import numpy as np
import torch
import os
import fnmatch
import json
import warnings
import importlib
from loguru import logger
import torch
import torch.distributed as dist
try:
    import pandas as pd
except:
    pd = None
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_wrappers import WrappedDataset, WrappedIterableDataset, MapToIterableDataset
from .normalize import NORMTYPE2CLASS, load_normalizers, save_norm_meta_to_json, load_normalizer_from_meta


def save_example_data(train_data, output_dir):
    """
    Save example data from the first dataset for debugging purposes.
    
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
        
        # Get one sample from the dataset
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
        
        # Save images - save each camera view separately
        if 'image' in sample and sample['image'] is not None:
            image_data = sample['image']
            
            # Convert tensor to numpy if needed
            if isinstance(image_data, torch.Tensor):
                image_data = image_data.cpu().numpy()
            
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
        
        # Save state and action as CSV
        state_action_data = {}
        
        if 'state' in sample and sample['state'] is not None:
            state = sample['state']
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            # Flatten if needed
            state = state.flatten()
            state_action_data['state'] = state
        
        if 'action' in sample and sample['action'] is not None:
            action = sample['action']
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            # Action might be (chunk_size, action_dim), save each timestep
            if len(action.shape) == 2:
                # Save as multiple rows
                csv_file = os.path.join(examples_dir, 'state_action.csv')
                df_data = {'timestep': list(range(action.shape[0]))}
                
                # Add action columns
                for i in range(action.shape[1]):
                    df_data[f'action_{i}'] = action[:, i]
                
                # Add state columns (broadcast state to all timesteps)
                if 'state' in state_action_data:
                    state = state_action_data['state']
                    for i in range(len(state)):
                        df_data[f'state_{i}'] = [state[i]] * action.shape[0]
                
                df = pd.DataFrame(df_data)
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved state and action (action shape: {action.shape}) to: {csv_file}")
            else:
                # Single action vector
                action = action.flatten()
                state_action_data['action'] = action
                
                # Create DataFrame
                csv_file = os.path.join(examples_dir, 'state_action.csv')
                df = pd.DataFrame([state_action_data])
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved state and action to: {csv_file}")
        
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
        
        logger.info("Successfully saved example data from first dataset")
        
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
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    
    # Create combined dataset
    train_data = wrapped_datasets[0] if len(wrapped_datasets) == 1 else wrapped_datasets
    return {'train': train_data, 'eval': None}

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