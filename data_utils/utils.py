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
    
def load_data(args, task_config, save_norm=True):
    dataset_dir_l = task_config['dataset_dir']
    # episode_len = task_config['episode_len']
    camera_names = task_config.get('camera_names', [])
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 1.0)
    ctrl_space = task_config.get('ctrl_space', 'ee')
    ctrl_type = task_config.get('ctrl_type', 'delta')
    data_class = task_config.get('dataset_class', 'EpisodicDataset')
    # Note: is_h5 parameter is deprecated. Dataset classes now automatically handle .h5 file traversal internally.
    action_normtype = args.action_normalize
    state_normtype = args.state_normalize
    if type(dataset_dir_l) == str: dataset_dir_l = [dataset_dir_l]
    if data_class == 'EpisodicDataset':
        from data_utils.datasets import EpisodicDataset
        data_class = EpisodicDataset
    else:
        data_class = getattr(importlib.import_module('data_utils.datasets'), data_class)
    # Compute statistics for each dataset
    # Pass directory path directly to dataset constructor, let dataset internally handle .h5 file traversal
    rank = dist.get_rank() if is_distributed() else 0
    if rank == 0:
        datasets = [data_class([dataset_dir], camera_names, data_args=args, chunk_size=args.chunk_size, ctrl_space=ctrl_space, ctrl_type=ctrl_type) for dataset_dir in dataset_dir_l]
    if is_distributed():
        dist.barrier()
    if rank != 0:
        datasets = [data_class([dataset_dir], camera_names, data_args=args, chunk_size=args.chunk_size, ctrl_space=ctrl_space, ctrl_type=ctrl_type) for dataset_dir in dataset_dir_l]
    # Get normalizer class
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    # Compute dataset statistics, dataset can select normalizer based on the dataset_dir of h5 files
    action_normalizers = {dataset.get_dataset_dir(): action_normalizer_class(dataset) for dataset in datasets}
    state_normalizers = {dataset.get_dataset_dir(): state_normalizer_class(dataset) for dataset in datasets}
    if save_norm:
        norm_meta = {'state': {k:str(v) for k,v in state_normalizers.items()}, 'action': {k:str(v) for k,v in action_normalizers.items()}, 'kwargs':{'ctrl_space':ctrl_space, 'ctrl_type':ctrl_type}}
        save_norm_meta_to_json(os.path.join(args.output_dir, 'normalize.json'), norm_meta)
        for dataset_dir_l, normalizer_l in state_normalizers.items():
            try:
                normalizer_l.save_stats_to_(args.output_dir)
            except Exception as e:
                print("Failed to save normalizer stats of {} because {}".format(dataset_dir_l, e))
    for dataset in datasets:
        dataset.set_action_normalizers(action_normalizers)
        dataset.set_state_normalizers(state_normalizers)
        
    train_dataset = ConcatDataset(datasets)
    x = train_dataset[0] # test __getitem__
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