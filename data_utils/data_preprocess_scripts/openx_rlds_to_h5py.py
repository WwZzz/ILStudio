import itertools
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os.path as osp
import copy
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from tqdm import tqdm
import json
import h5py
import tensorflow as tf
import copy
import torch
import collections
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import time
import argparse
from data_utils.rotate import *
from prismatic.vla.datasets.datasets import RLDSDataset, EpisodicRLDSDataset

def generate_h5(obs_replay, action_replay, language_raw, dataset_path, data_name, episode_id, camera_names=['primary'], img_size=(224,224), state_dim=7, action_dim=7):
    data_dict = {
        '/observations/qpos': obs_replay['qpos'],
        '/observations/qvel': obs_replay['qvel'],
        '/action': action_replay,
        '/episode_id': episode_id,
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = obs_replay['images'][cam_name]
    max_timesteps = len(data_dict['/observations/qpos'])
    # save the data, 2GB cache
    root = h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) 
    root.attrs['sim'] = False
    obs = root.create_group('observations')
    image = obs.create_group('images')
    for cam_name in camera_names:
        _ = image.create_dataset(cam_name, (max_timesteps,img_size[0], img_size[1], 3), dtype='uint8',  chunks=(1, img_size[0], img_size[1], 3), )
    qpos = obs.create_dataset('qpos', (max_timesteps, state_dim))
    qvel = obs.create_dataset('qvel', (max_timesteps, state_dim))
    action = root.create_dataset('action', (max_timesteps, action_dim))
    root.create_dataset("language_raw", data=[language_raw.encode('utf-8')])
    root.create_dataset("dataset", data=[data_name.encode('utf-8')])
    episode_data = root.create_dataset('episode_id', (1))
    for name, array in data_dict.items():
        root[name][...] = array

def ndarray_to_list(d):
    """
    递归地将字典中值为np.ndarray的元素转换为列表
    """
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            ndarray_to_list(value)
            
def denormalize_action(action, ds_stat):
    low = np.array(ds_stat['action']['min'])
    high = np.array(ds_stat['action']['max'])
    ori_action = 0.5*(action+1.)*(high-low+1e-8)+low
    return ori_action

parser = argparse.ArgumentParser()
# External config file that overwrites default config
parser.add_argument("--src_root", default= '/inspire/hdd/global_public/public_datas/Robotics_Related/Open-X-Embodiment/openx/')
parser.add_argument("--name", default="fmb")
# parser.add_argument(# "--target_root", default='.')
parser.add_argument("--target_root", default='/inspire/hdd/project/robot-action/public/data/openx_h5py')
args = parser.parse_args()
    
if __name__=='__main__':
    target_dir = args.target_root
    target_root = os.path.join(target_dir, args.name)
    os.makedirs(target_root, exist_ok=True)
    total_traj_cnt = 0
    # ds = RLDSDataset(args.src_root, args.name, None, (224, 224))
    wrapped_ds = EpisodicRLDSDataset(args.src_root, args.name, None, (224, 224)) # episode can be obtained by episode=next(iter(ds)), now the data has no 'qpos' in returns
    ds = wrapped_ds.dataset
    ds_stat = wrapped_ds.dataset_statistics
    ds_len = wrapped_ds.dataset_length
    counts=0
    if not os.path.exists(os.path.join(target_root, 'dataset_statistics.json')):
        ds_stat_copy = copy.deepcopy(ds_stat)
        with open(os.path.join(target_root, 'dataset_statistics.json'), 'w') as f:
            json.dump(ndarray_to_list(ds_stat_copy), f)
    for eidx, episode in tqdm(enumerate(ds), total=ds_len):
        target_path = os.path.join(target_root, f'episode_{total_traj_cnt}.hdf5')
        if os.path.exists(target_path):
            total_traj_cnt += 1
            continue
        language_raw = episode['task']['language_instruction'].numpy()[0].decode()
        actions = episode['action'][()].numpy().squeeze(1)
        actions = denormalize_action(actions, ds_stat)
        states = np.zeros_like(actions)
        images = episode['observation']['image_primary'].numpy().squeeze(1)
        obs_replay = {
            'qpos': states,
            'qvel': states,
            'images': {'primary': images,}
        }
        generate_h5(obs_replay, actions, language_raw, target_path, args.name, eidx)
        total_traj_cnt += 1
        # if total_traj_cnt>5: break