import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import IPython
import copy
e = IPython.embed
from configuration.utils import *
import gc
from .statistic import MinMaxNormalizer, PercentileNormalizer, ZScoreNormalizer

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, action_normalizers=None, state_normalizers=None, data_args=None, control_space='ee'):
        super(EpisodicDataset).__init__()
        self.episode_ids = np.arange(len(dataset_path_list))
        self.dataset_path_list = dataset_path_list
        self.action_normalizers = action_normalizers
        self.state_normalizers = state_normalizers
        self.camera_names = camera_names
        self.episode_len = get_episode_len(dataset_path_list)
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(self.episode_len)
        self.data_args = data_args
        self.control_space = control_space
        self.augment_images = False
        self.transformations = None
        # a = self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def load_from_h5(self, dataset_path, start_ts, episode_len): 
        with h5py.File(dataset_path, 'r') as root:
            # 加载文本
            raw_lang = root['language_instruction'][0].decode('utf-8')
            # 加载动作 & 状态
            action_start = max(0, start_ts - 1)
            action = root[f'/action_{self.control_space}'][action_start:action_start+self.data_args.chunk_size]
            if self.data_args.abs_control:
                states = root[f'/observations/state_{self.control_space}'][action_start:action_start+self.data_args.chunk_size]
                action[:, :-1] = action[:, :-1] + states[:, :-1]
                state = states[0]
            else:
                state = root[f'/observations/state_{self.control_space}'][start_ts]
            # 加载图像
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/image/{cam_name}'][start_ts]
                img_size = self.data_args.image_size_primary if 'primary' in cam_name else self.data_args.image_size_wrist
                image_dict[cam_name] = cv2.resize(image_dict[cam_name], eval(img_size))
            # 加载推理信息
            reasoning = ""
            if self.data_args.use_reasoning:
                if 'substep_reasonings' in root.keys(): 
                    reasoning = root['substep_reasonings'][start_ts].decode('utf-8')
                elif 'reasoning' in root.keys():
                    # construct reasoning
                    gpos = root['/reasoning/gripper_position'][start_ts]
                    tpos = root['/reasoning/target_position'][start_ts]
                    subtask = root['/reasoning/subtask'][start_ts].decode('utf-8')
                    prev_task = 'Init' if start_ts==0 else root['subtask'][start_ts-1].decode('utf-8')
                    drct = root['/reasoning/direction'][start_ts].decode('utf-8')
                    reasoning = f"Step: {subtask}\nGripper Pos: ({gpos[0]:.2f},{gpos[1]:.2f})\nTarget Pos: ({tpos[0]:.2f},{tpos[1]:.2f})\nDirection: {drct}"
                    if self.data_args.use_prev_subtask: raw_lang = raw_lang + f". The previous step is: {prev_task}"
                else:
                    try:
                        reasoning = root['reasoning'][0].decode('utf-8')
                    except Exception as e:
                        print(f"Read reasoning from {dataset_path} happens {YELLOW}{e}{RESET}")
                        exit(0)
        return action, image_dict, state, raw_lang, reasoning
    
    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        episode_len = self.episode_len[episode_id]
        try:
            action, image_dict, state, raw_lang, reasoning = self.load_from_h5(dataset_path, start_ts, episode_len)
        except Exception as e:
            print(f"Read {dataset_path} happens {YELLOW}{e}{RESET}")
            try:
                dataset_path = self.dataset_path_list[episode_id + 1]
            except Exception as e:
                dataset_path = self.dataset_path_list[episode_id - 1]
            action, image_dict, state, raw_lang, reasoning = self.load_from_h5(dataset_path, start_ts, episode_len)
        padded_action = np.zeros((self.data_args.chunk_size, action.shape[1]), dtype=np.float32) # padding动作到完整的维度
        padded_action[:action.shape[0]] = action
        is_pad = np.zeros(self.data_args.chunk_size) # 标注哪些位置的动作是padding的，不加入计算
        is_pad[action.shape[0]:] = 1
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0) #把img叠成一个array
        
        # normalize data
        action_normalizer = self.action_normalizers.get(os.path.dirname(dataset_path), None)
        assert action_normalizer is not None, f"no normalizer found for dataset dir {os.path.dirname(dataset_path)}"
        action_data = action_normalizer.normalize(padded_action, space_name=self.control_space, datatype='action', is_delta=not self.data_args.abs_control)
        state_normalizer = self.state_normalizers.get(os.path.dirname(dataset_path), None)
        assert state_normalizer is not None, f"no normalizer found for dataset dir {os.path.dirname(dataset_path)}"
        state_data = state_normalizer.normalize(state, space_name=self.control_space, datatype='state', is_delta=False)
        
        # construct observations， 把array转成tensor
        image_data = torch.from_numpy(all_cam_images)
        state_data = torch.from_numpy(state_data).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        image_data = torch.einsum('k h w c -> k c h w', image_data) # 把图像交换通道

        sample = {
            'image': image_data,
            'state': state_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning
        } # 构造样本dict
        assert raw_lang is not None, ""
        del image_data
        del state_data
        del action_data
        del is_pad
        del raw_lang
        del reasoning
        gc.collect()
        torch.cuda.empty_cache()
        return sample



def get_episode_len(dataset_path_list, rank0_print=print):
    all_episode_len = []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                elen = root['/episode_len'][0].astype(np.int32) if '/episode_len' in root else root['/action'][()].shape[0]
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_episode_len')
            rank0_print(e)
            quit()
        all_episode_len.append(elen) 
    return all_episode_len

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

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(args, task_config, rank0_print=print):
    set_seed(0)
    dataset_dir_l = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir_l = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 1.0)
    name_filter = task_config.get('name_filter', lambda n: n.endswith('hdf5'))
    action_normtype = task_config.get('action_normtype', 'minmax')
    state_normtype = task_config.get('state_normtype', 'minmax')
    batch_size_train, batch_size_val = args.per_device_train_batch_size, args.per_device_eval_batch_size
    skip_mirrored_data=args.skip_mirrored_data
    if type(dataset_dir_l) == str: dataset_dir_l = [dataset_dir_l]
    # 获取normalizer class
    NORMTYPE2CLASS = {
        'minmax': MinMaxNormalizer,
        'percentile': PercentileNormalizer, 
        'zscore': ZScoreNormalizer,
    }
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    # 计算数据集的统计量, 数据集内部可以根据h5文件所在dataset_dir来选择normalizer
    action_normalizers = {dataset_dir: action_normalizer_class(dataset_dir=dataset_dir) for dataset_dir in dataset_dir_l}
    state_normalizers = {dataset_dir: state_normalizer_class(dataset_dir=dataset_dir) for dataset_dir in dataset_dir_l}
    
    # find all data
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])# 第一个数据集的所有episode数量 
    dataset_path_list = flatten_list(dataset_path_list_list) # 展平所有数据集的episode
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)] #过滤非法文件
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list] #每个数据集的episode数量
    num_episodes_cumsum = np.cumsum(num_episodes_l) # 累加episode总数

    train_episode_ids = np.arange(num_episodes_cumsum)
    all_episode_len = get_episode_len(dataset_path_list)
    train_episode_len = all_episode_len
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, action_normalizers=action_normalizers, state_normalizers=state_normalizers,  data_args=args)
    x = train_dataset[0]
    val_dataset = None

    sampler_params = {
        'train': {"batch_size": batch_size_train, 'episode_len_l': train_dataset.episode_len, 'sample_weights':sample_weights, 'episode_first': args.episode_first},
        'eval': {"batch_size": batch_size_val, 'episode_len_l': [0], 'sample_weights': None, 'episode_first': args.episode_first} # unused
    }
    return train_dataset, val_dataset, sampler_params


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0

    return np.array([linear_vel, angular_vel])

### env utils

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
