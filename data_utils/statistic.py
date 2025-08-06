
# --- Mean-Variance Normalization (also known as Standardization or Z-Score Normalization) ---
#
# Description:
#   This method rescales the data to have a mean of 0 and a standard deviation of 1.
#   It is the most common normalization technique, especially for data that is
#   approximately Gaussian (normally distributed).
#
# Formula:
#   x_normalized = (x - mean) / (std_dev + epsilon)
#
# Where:
#   - `mean` is the mean of the training dataset.
#   - `std_dev` is the standard deviation of the training dataset.
#   - `epsilon` is a small constant (e.g., 1e-8) to prevent division by zero.
#
# Usage:
#   Typically, the mean and std_dev are pre-computed from a large, representative
#   offline dataset and then stored. These same values are used for normalization
#   during both training and inference.


# --- Percentile Normalization (Robust Normalization) ---
#
# Description:
#   This method is robust to outliers and non-Gaussian distributions. Instead of
#   using mean/std_dev, it normalizes based on percentile ranks within the data distribution.
#   It works by clipping the data to a specified percentile range and then scaling it
#   to a target interval, typically [-1, 1].
#
# A common implementation involves these steps:
#   1. Determine percentile boundaries from a representative dataset (e.g., 5th and 95th).
#   2. Clip incoming data to these boundaries.
#   3. Linearly scale the clipped value to the target range (e.g., [-1, 1]).
#
# Formula (for scaling to [-1, 1] using p5 and p95):
#   x_clipped = clip(x, p5, p95)
#   x_normalized = 2 * (x_clipped - p5) / (p95 - p5) - 1
#
# Where:
#   - `p5` is the 5th percentile of the dataset.
#   - `p95` is the 95th percentile of the dataset.
#
# Usage:
#   Excellent for actions or states with skewed distributions or extreme outliers,
#   such as force sensor readings or tool velocities.

# --- Min-Max Normalization ---
#
# Description:
#   This method linearly rescales the data to a fixed range, most commonly [0, 1] or [-1, 1].
#   It is useful when the upper and lower bounds of the data are known or fixed.
#
# Formula (for scaling to [0, 1]):
#   x_normalized = (x - min_val) / (max_val - min_val)
#
# Formula (for scaling to [-1, 1]):
#   x_normalized = 2 * (x - min_val) / (max_val - min_val) - 1
#
# Where:
#   - `min_val` is the minimum value of the dataset.
#   - `max_val` is the maximum value of the dataset.
#
# Usage:
#   Suitable for data with well-defined boundaries, such as robot joint angles (which have
#   physical limits) or image pixel intensities.
#   Note: This method is highly sensitive to outliers.

# 所以要计算的统计量包括：均值、标准差、最小值、最大值、百分位最小值、百分位最大值，有这些统计量之后，就可以计算归一化和反归一化了
# ----------------------- h5 structure
# # ACTION
# joint_action: (T, qpos_dim)
# ee_action: (T, ee_dim)
#
# # OBSERVATIONS
# observations:
# 	joint_pos: (T, qpos_dim)
# 	ee_pos: (T, ee_dim)
#   ...

import pickle
import numpy as np
import h5py
import os
from collections import defaultdict
from benchmark.base import MetaAction, MetaObs  
from typing import List
import fnmatch

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        if 'pointcloud' in root: continue
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    return hdf5_files

class BaseNormalizer:
    ALL_KEYS = ['/observations/state_joint', '/observations/state_ee', '/action_joint', '/action_ee'] # (T, dim) for each
    def __init__(self, dataset_dir:str=None, dataset_name:str=None, gripper_indices: List[int] = [-1],  *args, **kwargs):
        self.dataset_dir = dataset_dir # the directionary path of .hdf5 files
        self.dataset_name = dataset_name
        self.gripper_indices = gripper_indices
        self.stats_filename = 'dataset_stats.pkl' if dataset_name is None else f"{dataset_name}_dataset_stats.pkl"
        if self.is_stats_exist():
            self.all_stats = self.load_stats()
        else:
            self.all_stats = self.compute_and_save_stats()

    def is_stats_exist(self):
        return os.path.exists(os.path.join(self.dataset_dir, self.stats_filename))
    
    def load_data_from_h5(self, file_path):
        res_dict = {}
        with h5py.File(file_path, 'r') as root:
            for k in self.ALL_KEYS:
                if k in root:
                    res_dict[k] = root[k][()]
        return res_dict

    def compute_stats_for_array(self, data_k):
        return {
            "mean": data_k.mean(0).tolist(),
            "std": data_k.std(0).tolist(),
            "max": data_k.max(0).tolist(),
            "min": data_k.min(0).tolist(),
            "q01": np.quantile(data_k, 0.01, axis=0).tolist(),
            "q99": np.quantile(data_k, 0.99, axis=0).tolist(),
        }
    
    def compute_and_save_stats(self):
        all_h5_files = find_all_hdf5(self.dataset_dir, True)
        all_data = defaultdict(list)
        for each in all_h5_files:
            res_each = self.load_data_from_h5(each)
            for k in res_each: all_data[k].append(res_each[k])
        all_stats = {}
        num_transitions = -1
        all_stats['num_trajectories'] = len(all_h5_files)
        for k in all_data: all_data[k] = np.concatenate(all_data[k])
        for k in all_data:
            data_k = all_data[k]
            if 'num_transitions' not in all_stats: all_stats['num_transitions'] = data_k.shape[0]
            dict_k = {k.split('/')[-1]: self.compute_stats_for_array(data_k)}
            all_stats.update(dict_k)
        # compute absolute action's stats
        try:
            absolute_joint_action = all_data['/action_joint'] + all_data['/observations/state_joint']
            for gidx in self.gripper_indices:
                absolute_joint_action[:,gidx] = all_data['/action_joint'][:,gidx]
            all_stats.update({'action_joint_abs': self.compute_stats_for_array(absolute_joint_action)})
        except:
            print('Failed to compute stats for absolute action in joint space')
        try:
            absolute_ee_action = all_data['/action_ee'] + all_data['/observations/state_ee']
            for gidx in self.gripper_indices:
                absolute_ee_action[:, gidx] = all_data['/action_ee'][:,gidx]
            all_stats.update({'action_ee_abs': self.compute_stats_for_array(absolute_ee_action)})
        except:
            print('Failed to compute stats for delta action in ee space')    
        with open(os.path.join(self.dataset_dir, self.stats_filename), 'wb') as file:  # 使用二进制写模式 ('wb')
            pickle.dump(all_stats, file)
        return {k:{kk:np.array(vv) for kk,vv in v.items()} if isinstance(v, dict) else v for k,v in all_stats.items()}
    
    def save_stats(self, path: str):
        assert os.path.isdir(path), f"{path} must be directionary"
        with open(os.path.join(self.dataset_dir, self.stats_filename), 'wb') as file:  # 使用二进制写模式 ('wb')
            pickle.dump(all_stats, file)
        
    
    def load_stats(self):
        with open(os.path.join(self.dataset_dir, self.stats_filename), 'rb') as file:  
            all_stats = pickle.load(file)
        all_stats = {k:{kk:np.array(vv) for kk,vv in v.items()} if isinstance(v, dict) else v for k,v in all_stats.items()}
        return all_stats
    
    def get_stat_by_key(self, space_name='default', datatype='action', is_delta=True):
        assert datatype in ['action', 'state'], "datatype must be 'action' or 'state'"
        assert space_name in ['ee', 'joint'], "space_name must be in ['ee', 'joint']"
        if datatype=='action':
            if space_name=='ee': key='action_ee'
            else: key = 'action_joint'
            if not is_delta: key = key + '_abs'
        else:
            if space_name=='ee': key='state_ee'
            else: key = 'state_joint'
        if key not in self.all_stats: raise KeyError(f"Cannot find {key} in stats.")
        return self.all_stats[key]
    
    def normalize_metaobs(self, mobs: MetaObs, space_name='ee'):
        if mobs.state_ee is not None and (space_name=='ee' or space_name=='all'):
            mobs.state_ee = self.normalize(mobs.state_ee, space_name='ee', datatype='state')
            mobs.state = mobs.state_ee
        if mobs.state_joint is not None and (space_name=='joint' or space_name=='all'):
            mobs.state_joint = self.normalize(mobs.state_joint, space_name='joint', dataytpe='state')
            mobs.state = mobs.state_joint
        return mobs
    
    def denormalize_metaact(self, mact: MetaAction):
        mact.action = self.denormalize(mact.action, space_name=mact.space_name, datatype='action', is_delta=mact.is_delta)
        return mact
    
    def normalize(self, *args, **kwargs):
        raise NotImplementedError
    
    def denormalize(self, *args, **kwargs):
        raise NotImplementedError
    
    
class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, dataset_dir, dataset_name=None, low:float=-1, high:float=1):
        super().__init__(dataset_dir, dataset_name)
        assert low!=high, "low is equal to high"
        self.low = low
        self.high = high
        self.delta = self.high-self.low
        
    def __str__(self):
        return "minmax"
    
    def normalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = (data-stats['min'])/(stats['max'] - stats['min'])*self.delta+self.low
        return data.astype(dtype)
    
    def denormalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = ((data - self.low) / self.delta) * (stats['max'] - stats['min']) + stats['min']
        return data.astype(dtype)

class PercentileNormalizer(BaseNormalizer):
    def __init__(self, dataset_dir, dataset_name=None, low:float=-1, high:float=1):
        super().__init__(dataset_dir, dataset_name)
        assert low!=high, "low is equal to high"
        self.low = low
        self.high = high
        self.delta = self.high-self.low
    
    def __str__(self):
        return "percentile"
    
    def normalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = (data-stats['q01'])/(stats['q99'] - stats['q01'])*self.delta+self.low
        return np.clip(data, self.low, self.high).astype(dtype)
    
    def denormalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = ((data - self.low) / self.delta) * (stats['q99'] - stats['q01']) + stats['q01']
        return np.clip(data, stats['q01'], stats['q99']).astype(dtype)

class ZScoreNormalizer(BaseNormalizer):
    def __init__(self, dataset_dir, dataset_name=None, *args, **kwargs):
        super().__init__(dataset_dir, dataset_name)
        
    def __str__(self):
        return "zscore"
    
    def normalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = (data-stats['mean'])/stats['std'] 
        return data.astype(dtype)
    
    def denormalize(self, data, space_name='ee', datatype='action', is_delta=True):
        dtype = data.dtype
        stats = self.get_stat_by_key(space_name=space_name, datatype=datatype, is_delta=is_delta)
        data = data*stats['std'] + stats['mean']
        return data.astype(dtype)
    