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

def flatten_list(l):
    return [item for sublist in l for item in sublist]
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, robot=None, rank0_print=print, data_args=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.data_args = data_args
        self.robot = robot
        self.rank0_print = rank0_print
        self.augment_images = False
        self.transformations = None
        self.rank0_print(f"########################Current Image Size is [{self.data_args.image_size_stable}]###################################")
        a=self.__getitem__(0) # initialize self.is_sim and self.transformations
        if len(self.camera_names) > 2:
            # self.rank0_print("%"*40)
            self.rank0_print(f"The robot is {RED} {self.robot} {RESET} | The camera views: {RED} {self.camera_names} {RESET} | The history length: {RED} {self.data_args.history_images_length} {RESET}")
        self.is_sim = False

    def __len__(self):
        return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def load_from_h5(self, dataset_path, start_ts, episode_len): # 这里有一个问题，所有h5都是加载所有array后再取索引的，必然很慢；要改两个点：一个是h5最好在保存时就分块，另一个是每次只取指定的块
        with h5py.File(dataset_path, 'r') as root:
            try: # some legacy data does not have this attribute
                is_sim = root.attrs['sim']
            except:
                is_sim = False
            compressed = root.attrs.get('compress', False)
            if 'truncate' in dataset_path: compressed = False
            try:
                raw_lang = root['language_raw'][0].decode('utf-8')
            except Exception as e:
                # self.rank0_print(e)
                self.rank0_print(f"Read {dataset_path} happens {YELLOW}{e}{RESET}")
                exit(0)
            reasoning = " "
            if self.data_args.use_reasoning:
                if 'substep_reasonings' in root.keys(): 
                    reasoning = root['substep_reasonings'][start_ts].decode('utf-8')
                elif 'direction' in root.keys():
                    # construct reasoning
                    gpos = root['gripper_position'][start_ts]
                    tpos = root['target_position'][start_ts]
                    subtask = root['subtask'][start_ts].decode('utf-8')
                    prev_task = 'Init' if start_ts==0 else root['subtask'][start_ts-1].decode('utf-8')
                    drct = root['direction'][start_ts].decode('utf-8')
                    reasoning = f"Step: {subtask}\nGripper Pos: ({gpos[0]:.2f},{gpos[1]:.2f})\nTarget Pos: ({tpos[0]:.2f},{tpos[1]:.2f})\nDirection: {drct}"
                    if self.data_args.use_prev_subtask: 
                        raw_lang = raw_lang + f". The previous step is: {prev_task}"
                else:
                    try:
                        reasoning = root['reasoning'][0].decode('utf-8')
                    except Exception as e:
                        self.rank0_print(f"Read reasoning from {dataset_path} happens {YELLOW}{e}{RESET}")
                        exit(0)
            action_start = max(0, start_ts - 1)
            action = root['/action'][action_start:action_start+self.chunk_size]
            original_action_shape = action.shape
            action_len = action.shape[0]
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                image_dict[cam_name] = cv2.resize(image_dict[cam_name], eval(self.data_args.image_size_stable))
            if compressed:
                print(f"{RED} It's compressed in {dataset_path} {RESET}")
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)
        return original_action_shape, action, action_len, image_dict, qpos, qvel, raw_lang, reasoning
    
    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        episode_len = self.episode_len[episode_id]
        try:
            original_action_shape, action, action_len, image_dict, qpos, qvel, raw_lang, reasoning = self.load_from_h5(dataset_path, start_ts, episode_len)
        except Exception as e:
            print(f"Read {dataset_path} happens {YELLOW}{e}{RESET}")
            try:
                dataset_path = self.dataset_path_list[episode_id + 1]
            except Exception as e:
                dataset_path = self.dataset_path_list[episode_id - 1]
            original_action_shape, action, action_len, image_dict, qpos, qvel, raw_lang, reasoning = self.load_from_h5(dataset_path, start_ts, episode_len)
        padded_action = np.zeros((self.chunk_size, original_action_shape[1]), dtype=np.float32) # padding动作到完整的维度
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size) # 标注哪些位置的动作是padding的，不加入计算
        is_pad[action_len:] = 1
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0) #把img叠成一个array
        # construct observations， 把array转成tensor
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        image_data = torch.einsum('k h w c -> k c h w', image_data) # 把图像交换通道
        norm_stats = self.norm_stats 
        action_data = ((action_data - norm_stats["action_min"]) / (norm_stats["action_max"] - norm_stats["action_min"])) * 2 - 1 #动作归一化
        qpos_data = (qpos_data - norm_stats["qpos_mean"]) / norm_stats["qpos_std"] # state归一化
        sample = {
            'image': image_data,
            'state': qpos_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning
        } # 构造样本dict
        assert raw_lang is not None, ""
        if index == 0:
            self.rank0_print(reasoning)
        del image_data
        del qpos_data
        del action_data
        del is_pad
        del raw_lang
        del reasoning
        gc.collect()
        torch.cuda.empty_cache()
        return sample

# 这个函数用来预训练肯定是不对头的
def get_norm_stats(dataset_path_list, rank0_print=print):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
        except Exception as e:
            rank0_print(f'Error loading {dataset_path} in get_norm_stats')
            rank0_print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos)) # 所有数据集统一计算均值
        all_action_data.append(torch.from_numpy(action)) # 所有数据集统一计算方差
        all_episode_len.append(len(qpos)) 
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # 只对qpos和action作归一化，这里只计算动作的均值、方差、最大、最小值；和qpos的均值、方差；
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}
    return stats, all_episode_len

# calculating the norm stats corresponding to each kind of task (e.g. folding shirt, clean table....)
def get_norm_stats_by_tasks(dataset_path_list):

    data_tasks_dict = dict(
        fold_shirt=[],
        clean_table=[],
        others=[],
    )
    for dataset_path in dataset_path_list:
        if 'fold' in dataset_path or 'shirt' in dataset_path:
            key = 'fold_shirt'
        elif 'clean_table' in dataset_path and 'pick' not in dataset_path:
            key = 'clean_table'
        else:
            key = 'others'
        data_tasks_dict[key].append(dataset_path)

    norm_stats_tasks = {k : None for k in data_tasks_dict.keys()}

    for k,v in data_tasks_dict.items():
        if len(v) > 0:
            norm_stats_tasks[k], _ = get_norm_stats(v)

    return norm_stats_tasks


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
    batch_size_train, batch_size_val = args.per_device_train_batch_size, args.per_device_eval_batch_size
    skip_mirrored_data=args.skip_mirrored_data
    if type(dataset_dir_l) == str: dataset_dir_l = [dataset_dir_l]
    # find all data
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])# 第一个数据集的所有episode数量 
    dataset_path_list = flatten_list(dataset_path_list_list) # 展平所有数据集的episode
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)] #过滤非法文件
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list] #每个数据集的episode数量
    num_episodes_cumsum = np.cumsum(num_episodes_l) # 累加episode总数

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0) #
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0] # 只有第一个数据集的一部分作为验证集，主要是因为具身里验证集没意义

    train_episode_ids = np.concatenate(train_episode_ids_l) # 计算所有训练episode数量
    val_episode_ids = np.concatenate(val_episode_ids_l) # 计算所有验证episode数量
    # rank0_print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    all_episode_len = []
    all_stats = []
    for dataset_dir in dataset_dir_l:
        stats_path = os.path.join(dataset_dir, f'dataset_stats.pkl')
        len_path = os.path.join(dataset_dir, f'dataset_lens.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                crt_norm_stats = pickle.load(f)
            with open(len_path, 'rb') as f:
                crt_episode_len = pickle.load(f)
        else:
            crt_norm_stats, crt_episode_len = get_norm_stats(find_all_hdf5(dataset_dir, skip_mirrored_data))
            with open(stats_path, 'wb') as f:
                pickle.dump(crt_norm_stats, f)
            with open(len_path, 'wb') as f:
                pickle.dump(crt_episode_len, f)
        all_episode_len.append(crt_episode_len)
        all_stats.append(crt_norm_stats)
    all_episode_len = flatten_list(all_episode_len)
    
    # _, all_episode_len = get_norm_stats(dataset_path_list) #获取每个数据集的stat，和所有episode的长度
    # rank0_print(f"{RED}All images: {sum(all_episode_len)}, Trajectories: {len(all_episode_len)} {RESET}")
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l] # 获取每个数据集的训练集的episode长度
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l] # 获取每个数据集的验证集的episode长度

    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    # if stats_dir_l is None:
    #     stats_dir_l = dataset_dir_l
    # elif type(stats_dir_l) == str:
    #     stats_dir_l = [stats_dir_l]
    
    norm_stats  = {}
    all_weights = np.array([num_episodes_l]).T
    total = all_weights.sum()
    for k in all_stats[0].keys():
        all_stats_k = np.stack([si[k] for si in all_stats])
        if 'mean' in k:
            norm_stats[k] = (all_stats_k*all_weights).sum(axis=0)/total
        elif 'min' in k:
            norm_stats[k] = np.min(all_stats_k, axis=0)
        elif 'max' in k:
            norm_stats[k] = np.max(all_stats_k, axis=0)
        elif 'std' in k:
            mean_k = k.replace('std', 'mean')
            mean_values = np.stack([si[mean_k] for si in all_stats])
            mean_values_2 = mean_values**2
            std_values_2 = all_stats_k**2
            var = (mean_values_2+std_values_2)*all_weights.sum(axis=0)/total-norm_stats[mean_k]**2
            norm_stats[k] = np.sqrt(var)
        else:
            continue
    # 计算所有episode的stat
    # norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    chunk_size = args.chunk_size
    # calculate norm stats corresponding to each kind of task
    # rank0_print(f'Norm stats from: {[each.split("/")[-1] for each in stats_dir_l]}')
    # rank0_print(f'train_episode_len_l: {train_episode_len_l}')

    robot = 'franka'
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size,  robot=robot,  data_args=args)
    # val_dataset is unused
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size,  robot=robot, data_args=args) if sum(val_episode_len)>0 else None

    sampler_params = {
        'train': {"batch_size": batch_size_train, 'episode_len_l': train_episode_len_l, 'sample_weights':sample_weights, 'episode_first': args.episode_first},
        'eval': {"batch_size": batch_size_val, 'episode_len_l': val_episode_len_l, 'sample_weights': None, 'episode_first': args.episode_first} # unused
    }
    stats_path = os.path.join(args.output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)
    return train_dataset, val_dataset, norm_stats, sampler_params


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
