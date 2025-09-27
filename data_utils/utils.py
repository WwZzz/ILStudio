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

# e = IPython.embed  # Removed to avoid unnecessary dependency

# Normalize Class
NORMTYPE2CLASS = {
    'minmax': MinMaxNormalizer,
    'percentile': PercentileNormalizer, 
    'zscore': ZScoreNormalizer,
    'identity': Identity,
}



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


# def smooth_base_action(base_action):
#     return np.stack([
#         np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
#     ], axis=-1).astype(np.float32)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list:list, camera_names:list, action_normalizers:dict={}, state_normalizers:dict={}, data_args=None, chunk_size:int=16, ctrl_space: str='ee', ctrl_type: str='delta'):
        super(EpisodicDataset).__init__()
        self.episode_ids = np.arange(len(dataset_path_list))
        self.dataset_path_list = dataset_path_list
        self.action_normalizers = action_normalizers
        self.state_normalizers = state_normalizers
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.data_args = data_args
        self.ctrl_space = ctrl_space # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type # ['abs', 'rel', 'delta']
        self.freq = -1
        self.max_workers = 8
        self.initialize()
    
    def initialize(self):
        self.loaded_data = self._load_all_episodes_into_memory() if getattr(self.data_args, 'preload_data', False) else None
        self.episode_len = self.get_episode_len() # 获取每个episode的长度
        self.cumulative_len = np.cumsum(self.episode_len) # 计算所有episode按顺序的累加长度
        self.max_episode_len = max(self.episode_len) # 统计最大episode长度
    
    def _load_file_into_memory(self, dataset_path):
        """
        加载一个 HDF5 文件，并扁平化其内容
        """
        flattened_data = {}
        with h5py.File(dataset_path, 'r') as f:
            def recursive_load(group, current_path=""):
                for key, item in group.items():
                    full_path = f"{current_path}/{key}" if current_path else f"/{key}"
                    if isinstance(item, h5py.Group):
                        recursive_load(item, full_path)
                    elif isinstance(item, h5py.Dataset):
                        flattened_data[full_path] = item[()]
            recursive_load(f)
        return {dataset_path: flattened_data}
    
    def _load_all_episodes_into_memory(self):
        """
        并行加载所有 HDF5 文件
        """
        print("Pre-Loading all data into memory...")
        memory_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交并行任务
            results = executor.map(self._load_file_into_memory, self.dataset_path_list)
            # 收集结果
            for result in results:
                memory_data.update(result)
        print("Pre-Loading Done")
        return memory_data

    def get_episode_len(self):
        if self.loaded_data is not None:
            tmp = self.loaded_data[list(self.loaded_data.keys())[0]]
            all_ks = ['/action', '/actions', '/action_ee', '/action_joint', '/state']
            key = None
            for k in all_ks:
                if k in tmp:
                    key = k
                    break
            if key is None: raise NotImplementedError("Failed to get length of episodes")
            all_episode_len = [
                self.loaded_data[pi]['/episode_len'][0].astype(int) 
                if '/episode_len' in self.loaded_data[pi]
                else self.loaded_data[pi][key].shape[0] 
                for pi in self.dataset_path_list
            ]
        else:
            all_episode_len = []
            key = None
            for dataset_path in self.dataset_path_list:
                try:
                    with h5py.File(dataset_path, 'r') as root:
                        if key is None:
                            all_ks = ['/action', '/actions', '/action_ee', '/action_joint', '/state']
                            for k in all_ks:
                                if k in root:
                                    key = k
                                    break
                        elen = root['/episode_len'][0].astype(np.int32) if '/episode_len' in root else root[key][()].shape[0]
                except Exception as e:
                    print(f'Error loading {dataset_path} in get_episode_len')
                    quit()
                all_episode_len.append(elen) 
        return all_episode_len
    
    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        # 把样本索引转换成episode的索引，和内部的时间步index
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def set_action_normalizers(self, ns):
        # 设置动作normalizers
        self.action_normalizers = ns
    
    def get_freq(self):
        return self.freq
    
    def set_state_normalizers(self, ns):
        # 设置状态normalizers
        self.state_normalizers = ns
    
    def get_episode_path(self, idx):
        return self.dataset_path_list[idx]
    
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load one-step data at start_ts from the episode specified by dataset_path"""
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # 加载文本
        raw_lang = root['language_instruction'][0].decode('utf-8')
        # 加载动作 & 状态
        action = root[f'/action_{self.ctrl_space}'][start_ts:start_ts+self.chunk_size]
        # 根据控制类型加载相应动作数据
        if self.ctrl_type=='abs':
            states = root[f'/observations/state_{self.ctrl_space}'][start_ts:start_ts+self.chunk_size]
            state = states[0]
            action[:, :-1] = action[:, :-1] + states[:, :-1]
        elif self.ctrl_type=='delta':
            state = root[f'/observations/state_{self.ctrl_space}'][start_ts]
        elif self.ctrl_type=='rel':
            raise NotImplementedError("relative action was not implemented")
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
                prev_task = 'Init' if start_ts==0 else root['/reasoning/subtask'][start_ts-1].decode('utf-8')
                drct = root['/reasoning/direction'][start_ts].decode('utf-8')
                reasoning = {
                    'subtask': subtask,
                    'gripper_pos': gpos,
                    'target pos': tpos,
                    'direction': drct,
                }
                # reasoning = f"Step: {subtask}\nGripper Pos: ({gpos[0]:.2f},{gpos[1]:.2f})\nTarget Pos: ({tpos[0]:.2f},{tpos[1]:.2f})\nDirection: {drct}"
                if self.data_args.use_prev_subtask: raw_lang = raw_lang + f". The previous step is: {prev_task}"
            else:
                try:
                    reasoning = root['reasoning'][start_ts].decode('utf-8')
                except Exception as e:
                    print(f"Read reasoning from {dataset_path} happens {e}")
                    exit(0)
        if self.loaded_data is None: root.close()
        return {
            'action': action,
            'image': image_dict,
            'state': state,
            'language_instruction': raw_lang,
            'reasoning': reasoning,
        }
        
    def load_feat_from_episode(self, dataset_path, feats=[]):
        """Load all steps data from the episode specified by dataset_path"""
        data_dict = {}
        if isinstance(feats, str): feats = [feats]
        with h5py.File(dataset_path, 'r') as root:
            if 'language_instruction' in feats or len(feats)==0: data_dict['language_instruction'] = root['language_instruction'][0].decode('utf-8') # 加载文本
            if 'state' in feats or len(feats)==0: data_dict['state'] = root[f'/observations/state_{self.ctrl_space}'][()] # 加载状态
            if 'action' in feats or len(feats)==0: # 加载动作 
                data_dict['action'] = root[f'/action_{self.ctrl_space}'][()] # 根据控制类型加载相应动作数据
                if self.ctrl_type=='abs': data_dict['action'] = data_dict['action'] + data_dict.get('state', root[f'/observations/state_{self.ctrl_space}'][()])
                elif self.ctrl_type=='rel':
                    raise NotImplementedError("relative action was not implemented")
            if 'image' in feats or len(feats)==0: # 加载图像
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/image/{cam_name}'][()]
                    # img_size = self.data_args.image_size_primary if 'primary' in cam_name else self.data_args.image_size_wrist
                    # image_dict[cam_name] = cv2.resize(image_dict[cam_name], eval(img_size))
                data_dict['image'] = image_dict
            reasoning = ""
            if 'reasoning' in feats or len(feats)==0: # 加载推理信息
                if 'substep_reasonings' in root.keys(): 
                    reasoning = root['substep_reasonings'][()].decode('utf-8')
                elif 'reasoning' in root.keys():
                    # construct reasoning
                    gpos = root['/reasoning/gripper_position'][()]
                    tpos = root['/reasoning/target_position'][()]
                    subtask = root['/reasoning/subtask'][()].decode('utf-8')
                    drct = root['/reasoning/direction'][()].decode('utf-8')
                    reasoning = {
                        'subtask': subtask,
                        'gripper_pos': gpos,
                        'target pos': tpos,
                        'direction': drct,
                    }
                else:
                    try:
                        reasoning = root['reasoning'][()].decode('utf-8')
                    except Exception as e:
                        print(f"Read reasoning from {dataset_path} happens {e}")
                        exit(0)
        return data_dict

    def extract_from_episode(self, episode_idx, keyname=[]):
        episode_path = self.dataset_path_list[episode_idx]
        feat = self.load_feat_from_episode(episode_path, keyname)
        return feat
    
    def get_dataset_dir(self):
        return os.path.dirname(self.dataset_path_list[0])
    
    @property
    def num_episodes(self):
        return len(self.dataset_path_list)
    
    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index) # 把样本索引转换成episode的索引，和内部的时间步index
        dataset_path = self.dataset_path_list[episode_id] # 从episode索引获取相应的episode路径
        episode_len = self.episode_len[episode_id] # 获得episode长度
        
        ############################# 加载数据 #################################
        data_dict = self.load_onestep_from_episode(dataset_path, start_ts) # 从文件和时间步加载数据
        action, image_dict, state, raw_lang = data_dict['action'], data_dict['image'], data_dict['state'], data_dict['language_instruction']
        reasoning = data_dict.get('reasoning', '')
        # process action
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
        if action_normalizer is not None:
            action_data = action_normalizer.normalize(padded_action, datatype='action')
        else:
            action_data = padded_action
            warnings.warn("No Normalization being applied to actions during training")
        state_normalizer = self.state_normalizers.get(os.path.dirname(dataset_path), None)
        if state_normalizer is not None:
            state_data = state_normalizer.normalize(state, datatype='state')
        else:
            state_data = state
            warnings.warn("No Normalization being applied to states during training")
        # construct observations， 把array转成tensor
        image_data = torch.from_numpy(all_cam_images)
        state_data = torch.from_numpy(state_data).float()
        action_data = torch.from_numpy(action_data).float()
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
        # gc.collect()
        # torch.cuda.empty_cache()
        return sample


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
    把 归一化加载信息 追加写入 json
    """
    # 如果文件不存在，先写一个空列表占位
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'state':{}, 'action':{}}, f)

    # 以 r+ 打开，先读后写
    with open(file_path, 'r+', encoding='utf-8') as f:
        # 读取已有内容
        try:
            old = json.load(f)
        except json.JSONDecodeError:
            # 文件为空或格式损坏，重新初始化
            old = {'state':{}, 'action':{}}

        # 移动指针到文件开头
        f.seek(0)
        # 追加新 dict
        old['state'].update(data.get('state', {}))
        old['action'].update(data.get('action', {}))
        old['kwargs'] = data.get('kwargs', {})
        # 写回
        json.dump(old, f, ensure_ascii=False, indent=2)
        # 截断多余内容（当新内容比旧内容短时）
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
    set_seed(0)
    dataset_dir_l = task_config['dataset_dir']
    # episode_len = task_config['episode_len']
    camera_names = task_config.get('camera_names', [])
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 1.0)
    ctrl_space = task_config.get('ctrl_space', 'ee')
    ctrl_type = task_config.get('ctrl_type', 'delta')
    data_class = task_config.get('dataset_class', 'EpisodicDataset')
    is_h5 = task_config.get('is_h5', True)
    action_normtype = args.action_normalize
    state_normtype = args.state_normalize
    if type(dataset_dir_l) == str: dataset_dir_l = [dataset_dir_l]
    if data_class == 'EpisodicDataset':
        from data_utils.datasets import EpisodicDataset
        data_class = EpisodicDataset
    else:
        data_class = getattr(importlib.import_module('data_utils.datasets'), data_class)
    # 以数据集为维度，计算统计量
    if is_h5:
        datasets = [data_class(find_all_hdf5(dataset_dir, True), camera_names, data_args=args, chunk_size=args.chunk_size, ctrl_space=ctrl_space, ctrl_type=ctrl_type) for dataset_dir in dataset_dir_l]
    else:
        datasets = [data_class([dataset_dir], camera_names, data_args=args, chunk_size=args.chunk_size, ctrl_space=ctrl_space, ctrl_type=ctrl_type) for dataset_dir in dataset_dir_l]
    # 获取normalizer class
    action_normalizer_class = NORMTYPE2CLASS[action_normtype]
    state_normalizer_class = NORMTYPE2CLASS[state_normtype]
    # 计算数据集的统计量, 数据集内部可以根据h5文件所在dataset_dir来选择normalizer
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