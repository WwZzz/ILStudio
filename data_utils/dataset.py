import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
import json
from time import time
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import IPython
from data_utils.rotate import quat2axisangle
from collections import OrderedDict
import copy
e = IPython.embed
from .utils import *

class AlohaSimDataset(EpisodicDataset):
    def get_language_instruction(self):
        task_name = os.path.split(self.get_dataset_dir())[-1]
        if 'transfer' in task_name:
            return 'Transfer the red cube to the other arm.'
        elif 'insert' in task_name:
            return 'Insert the red peg into the blue socket.'
        else:
            raise ValueError("Unknown task")
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load one-step data at start_ts from the episode specified by dataset_path"""
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # 加载文本
        raw_lang = self.get_language_instruction()
        # 加载动作 & 状态
        action = root[f'/action'][start_ts:start_ts+self.chunk_size]
        # 根据控制类型加载相应动作数据
        if self.ctrl_type=='abs':
            state = root[f'/observations/qpos'][start_ts]
        elif self.ctrl_type=='delta':
            states =  root[f'/observations/qpos'][start_ts:start_ts+self.chunk_size]
            action = action - states
            state = states[0]
        elif self.ctrl_type=='rel':
            raise NotImplementedError("relative action was not implemented")
        # 加载图像
        image_dict = dict(primary=cv2.resize(root['/observations/images/top'][start_ts], eval(self.data_args.image_size_primary)))
        if '/observations/images/left_wrist' in root:
            image_dict.update(
                dict(wrist_left=cv2.resize(root['/observations/images/left_wrist'][start_ts], eval(self.data_args.image_size_wrist)))
            )
        if '/observations/images/right_wrist' in root:
            image_dict.update(
                dict(wrist_right=cv2.resize(root['/observations/images/right_wrist'][start_ts], eval(self.data_args.image_size_wrist)))
            )
        # 加载推理信息
        reasoning = ""
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
            if 'language_instruction' in feats or len(feats)==0: data_dict['language_instruction'] = self.get_language_instruction() # 加载文本
            if 'state' in feats or len(feats)==0: data_dict['state'] = root[f'/observations/qpos'][()] # 加载状态
            if 'action' in feats or len(feats)==0: # 加载动作 
                data_dict['action'] = root[f'/action'][()] # 根据控制类型加载相应动作数据
                if self.ctrl_type=='delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type=='rel':
                    raise NotImplementedError("relative action was not implemented")
            if 'image' in feats or len(feats)==0: # 加载图像
                all_images = root[f'/observations/images/top'][()]
                image_dict = dict(primary=[cv2.resize(all_images[i], eval(self.data_args.image_size_primary)) for i in range(all_images.shape[0])])
                data_dict['image'] = image_dict
            if 'image' in feats or 'image_wrist' in feats or len(feats)==0:
                all_left_images = root[f'/observations/images/left_wrist'][()]
                left_dict = dict(wrist_left=[cv2.resize(all_left_images[i], eval(self.data_args.image_size_wrist)) for i in range(all_left_images.shape[0])])
                all_right_images = root[f'/observations/images/right_wrist'][()]
                right_dict = dict(wrist_right=[cv2.resize(all_right_images[i], eval(self.data_args.image_size_wrist)) for i in range(all_right_images.shape[0])])
                data_dict['image'].update(left_dict)
                data_dict['image'].update(right_dict)
        return data_dict

class AlohaSIIDataset(EpisodicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = 25 # 25hz 
    
    def get_freq(self):
        return self.freq
    
    def get_language_instruction(self):
        if 'red_cube' in self.get_dataset_dir():
            return "Pick up the red cube and put it on the top of the blue cup"
        return ""
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load one-step data at start_ts from the episode specified by dataset_path"""
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # 加载文本
        raw_lang = self.get_language_instruction()
        # 加载动作 & 状态
        action = root[f'/action'][start_ts:start_ts+self.chunk_size]
        # 根据控制类型加载相应动作数据
        if self.ctrl_type=='abs':
            state = root[f'/observations/qpos'][start_ts]
        elif self.ctrl_type=='delta':
            states = root[f'/observations/qpos'][start_ts:start_ts+self.chunk_size]
            action = action - states
            state = states[0]
        elif self.ctrl_type=='rel':
            raise NotImplementedError("relative action was not implemented")
        # 加载图像
        image_dict = dict(
            primary =  cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_high'][start_ts], np.uint8), cv2.IMREAD_COLOR), eval(self.data_args.image_size_primary)),
            wrist_left = cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_left_wrist'][start_ts], np.uint8), cv2.IMREAD_COLOR),eval(self.data_args.image_size_wrist)),
            wrist_right = cv2.resize(cv2.imdecode(np.frombuffer(root[f'/observations/images/cam_right_wrist'][start_ts], np.uint8), cv2.IMREAD_COLOR),eval(self.data_args.image_size_wrist)),
        )
        # 加载推理信息
        reasoning = ""
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
            if 'language_instruction' in feats or len(feats)==0: data_dict['language_instruction'] = self.get_language_instruction() # 加载文本
            if 'state' in feats or len(feats)==0: data_dict['state'] = root[f'/observations/qpos'][()] # 加载状态
            if 'qpos' in feats or len(feats)==0: data_dict['qpos'] = root[f'/observations/qpos'][()]
            if 'qvel' in feats or len(feats)==0: data_dict['qvel'] = root[f'/observations/qvel'][()]
            if 'action' in feats or len(feats)==0: # 加载动作 
                data_dict['action'] = root[f'/action'][()] # 根据控制类型加载相应动作数据
                if self.ctrl_type=='delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type=='rel':
                    raise NotImplementedError("relative action was not implemented")
            image_dict = dict()
            if 'image_primary' in feats or 'image' in feats or len(feats)==0: # 加载图像
                img_bytes = root[f'/observations/images/cam_high'][()]
                image_dict.update(
                    dict(
                        primary =  np.stack([cv2.resize(cv2.imdecode(np.frombuffer(img_byte, np.uint8), cv2.IMREAD_COLOR), eval(self.data_args.image_size_primary)) for img_byte in img_bytes]),
                    )
                )
            if 'image_wrist' in feats or 'image' in feats or len(feats)==0:
                left_bytes = root[f'/observations/images/cam_left_wrist'][()]
                image_dict.update(
                    dict(
                        wrist_left = np.stack([cv2.resize(cv2.imdecode(np.frombuffer(lb, np.uint8), cv2.IMREAD_COLOR),eval(self.data_args.image_size_wrist)) for lb in left_bytes]),
                    )
                )
                del left_bytes
                right_bytes = root[f'/observations/images/cam_right_wrist'][()]
                image_dict.update(
                    dict(
                        wrist_right = np.stack([cv2.resize(cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR),eval(self.data_args.image_size_wrist)) for rb in right_bytes]),
                    )
                )
                del right_bytes
            if len(image_dict)>0: data_dict['image'] = image_dict
        return data_dict
    
    def get_joint_limits(self):
        lower = np.array([-2.687807, 0.0, -3.054326, -1.850049, -1.308997, -1.745329])
        upper = np.array([2.687807, 3.403392, 0.0, 1.850049, 1.308997, 1.745329])
        return {'upper':upper, 'lower':lower}
    
class RobomimicDataset(EpisodicDataset):
    def initialize(self):
        from robomimic.utils.dataset import SequenceDataset
        self._datasets = [SequenceDataset(**self.create_config(di)) for di in self.dataset_path_list if 'image' in di]
        self._languages = [self.get_raw_lang(di) for di in self.dataset_path_list  if 'image' in di]
        self._dataset_dir = os.path.dirname(self.dataset_path_list[0])
        self.episode_ids = np.arange(sum(d.n_demos for d in self._datasets))
        self.dataset_path_list = sum([[f"{idx}:{ep}" for ep in di.demos] for idx,di in enumerate(self._datasets)], [])
        self.episode_len = self.get_episode_len() # 获取每个episode的长度
        self.cumulative_len = np.cumsum(self.episode_len) # 计算所有episode按顺序的累加长度
        self.max_episode_len = max(self.episode_len) # 统计最大episode长度

    def get_dataset_dir(self):
        return self._dataset_dir
    
    def get_raw_lang(self, data_path):
        # init raw lang
        from benchmark.robomimic import ALL_ENV_LANGUAGES
        if 'can' in data_path: return ALL_ENV_LANGUAGES['PickPlaceCan']
        elif 'lift' in data_path: return ALL_ENV_LANGUAGES['Lift']
        elif 'tool_hang' in data_path: return ALL_ENV_LANGUAGES['ToolHang']
        elif 'square' in data_path: return ALL_ENV_LANGUAGES['NutAssemblySquare']
        elif 'transport' in data_path: return ALL_ENV_LANGUAGES['TwoArmTransport']
        else:
            raise KeyError("Unknown language")
        
    
    def create_config(self, data_path, filter_by_attribute='train', seq_length=1):
        obs_key_shapes = OrderedDict([('agentview_image', [3, 84, 84]), ('robot0_eef_pos', [3]), ('robot0_eef_quat', [4]), ('robot0_eye_in_hand_image', [3, 84, 84]), ('robot0_gripper_qpos', [2])])
        return {
            'hdf5_path': data_path,
            'obs_keys': list(obs_key_shapes.keys()),
            'dataset_keys': ('actions', 'rewards', 'dones'),
            'load_next_obs': False,
            'frame_stack': 1,
            'seq_length': seq_length,
            'pad_frame_stack': True,
            'pad_seq_length': True,
            'get_pad_mask': False,
            'goal_mode': None,
            'hdf5_cache_mode': 'all' if getattr(self.data_args, 'preload_data else', True) else 'low_dim',
            'hdf5_use_swmr': True,
            'hdf5_normalize_obs': False,
            'filter_by_attribute': filter_by_attribute,
        }
    
    def get_episode_len(self):
        all_ep_lengths = []
        for dataset_path in self.dataset_path_list:
            idx, ep = dataset_path.split(':')
            ep_length = self._datasets[eval(idx)]._demo_id_to_demo_length[ep]
            all_ep_lengths.append(ep_length)
        return all_ep_lengths
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load one-step data at start_ts from the episode specified by dataset_path"""
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        demo_start_index = dataset._demo_id_to_start_indices[ep]  # demo 起始的全局索引
        global_index = demo_start_index + start_ts
        data = dataset[global_index]
        # load language 
        raw_lang = self._languages[eval(dataset_idx)]
        # load state
        state_euler = quat2axisangle(data['obs']['robot0_eef_quat'])
        state_xyz = data['obs']['robot0_eef_pos']
        gpos = data['obs']['robot0_gripper_qpos']
        state_gripper = (gpos[:,0]-gpos[:,1])[:,np.newaxis]
        state = np.concatenate([state_xyz, state_euler, state_gripper], axis=1)[0]
        # load action
        if dataset.hdf5_cache_mode=='all':
            demo_length = dataset._demo_id_to_demo_length[ep]
            chunk_size = min(self.chunk_size, demo_length-start_ts)
            action = np.concatenate([data['actions']]+[dataset[i]['actions'] for i in range(global_index+1, global_index+chunk_size)], axis=0)
        else:
            action = data['actions'] if self.chunk_size==1 else dataset.get_dataset_for_ep(ep=ep, key="actions")[start_ts:start_ts+self.chunk_size]
        action[:,:6] = action[:,:6]*np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        action[:,-1] = 0.5*(1-action[:,-1])
        if self.ctrl_type=='abs':
            if dataset.hdf5_cache_mode=='all':
                next_data = [data]+[dataset[i] for i in range(global_index+1, global_index+chunk_size)]
                xyzs = np.concatenate([di['obs']['robot0_eef_pos'] for di in next_data], axis=0)
                eulers = quat2axisangle(np.concatenate([di['obs']['robot0_eef_quat'] for di in next_data], axis=0))
                states = np.concatenate([xyzs, eulers], axis=1)
            else:
                xyzs = dataset.get_dataset_for_ep(ep=ep, key="obs/robot0_eef_pos")[start_ts:start_ts+self.chunk_size]
                eulers = quat2axisangle(dataset.get_dataset_for_ep(ep=ep, key="obs/robot0_eef_quat")[start_ts:start_ts+self.chunk_size]) 
                states = np.concatenate([xyzs, eulers], axis=1)
            action[:, :6] = action[:, :6] + states
        # load image
        image_dict = dict(
            primary=data['obs']['agentview_image'][0],
            wrist=data['obs']['robot0_eye_in_hand_image'][0],
        )
        return dict(
            action=action,
            state=state,
            image=image_dict,
            language_instruction=raw_lang,
            reasoning="",
        )
       
    
    def load_feat_from_episode(self, dataset_path, feats=[]):
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        if dataset.hdf5_cache_mode=='all':
            demo_start = dataset._demo_id_to_start_indices[ep]
            demo_length = dataset._demo_id_to_demo_length[ep]
            demo_end = demo_start + demo_length
            trajectory = {
                "obs": [dataset.getitem_cache[i]["obs"] for i in range(demo_start, demo_end)],
                "actions": [dataset.getitem_cache[i]["actions"] for i in range(demo_start, demo_end)],
            }
            trajectory["obs"] = {k: np.concatenate([obs[k] for obs in trajectory["obs"]], axis=0) for k in trajectory["obs"][0]}
            trajectory["actions"] = np.concatenate(trajectory["actions"], axis=0)
            trajectory_data = trajectory
        else:
            demo_index = dataset.demos.index(ep)
            trajectory_data = dataset.get_trajectory_at_index(demo_index)
        data_dict = {}
        if isinstance(feats, str): feats = [feats]
        if 'language_instruction' in feats or len(feats)==0: data_dict['language_instruction'] = self._languages[dataset_idx]
        if 'state' in feats or len(feats)==0: 
            state_euler = quat2axisangle(trajectory_data['obs']['robot0_eef_quat'])
            state_xyz = trajectory_data['obs']['robot0_eef_pos']
            gpos = trajectory_data['obs']['robot0_gripper_qpos']
            state_gripper = (gpos[:,0]-gpos[:,1])[:,np.newaxis]
            state = np.concatenate([state_xyz, state_euler, state_gripper], axis=1)
            data_dict['state'] = state
        if 'action' in feats or len(feats)==0: # 加载动作 
            action = trajectory_data['actions']
            action[:,:6] = action[:,:6]*np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
            action[:,-1] = 0.5*(1-action[:,-1])
            data_dict['action'] = action
        if 'image' in feats or len(feats)==0: # 加载图像
            image_dict = dict(
                primary=trajectory_data['obs']['agentview_image'],
                wrist=trajectory_data['obs']['robot0_eye_in_hand_image']
            )
            data_dict['image'] = image_dict
        return data_dict
    
    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id] 
        episode_len = self.episode_len[episode_id] 
        data_dict = self.load_onestep_from_episode(dataset_path, start_ts) 
        action, image_dict, state, raw_lang = data_dict['action'], data_dict['image'], data_dict['state'], data_dict['language_instruction']
        reasoning = data_dict.get('reasoning', '')
        padded_action = np.zeros((self.data_args.chunk_size, action.shape[1]), dtype=np.float32) 
        padded_action[:action.shape[0]] = action
        is_pad = np.zeros(self.data_args.chunk_size) 
        is_pad[action.shape[0]:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0) #把img叠成一个array
        # normalize data
        action_normalizer = self.action_normalizers.get(self.get_dataset_dir(), None)
        if action_normalizer is not None:
            action_data = action_normalizer.normalize(padded_action, datatype='action')
        else:
            action_data = padded_action
            warnings.warn("No Normalization being applied to actions during training")
        state_normalizer = self.state_normalizers.get(self.get_dataset_dir(), None)
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


class AlohaSIIv2Dataset(EpisodicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = 25 # 25hz 
    
    def get_freq(self):
        return self.freq
    
    def get_language_instruction(self):
        return "Put the green duck to the blue bowl and the orange square to the pink plate"
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """Load one-step data at start_ts from the episode specified by dataset_path"""
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # 加载文本
        raw_lang = self.get_language_instruction()
        # 加载动作 & 状态
        action = root[f'/action'][start_ts:start_ts+self.chunk_size]
        # 根据控制类型加载相应动作数据
        if self.ctrl_type=='abs':
            state = root[f'/observations/qpos'][start_ts]
        elif self.ctrl_type=='delta':
            states = root[f'/observations/qpos'][start_ts:start_ts+self.chunk_size]
            action = action - states
            state = states[0]
        elif self.ctrl_type=='rel':
            raise NotImplementedError("relative action was not implemented")
        # 加载图像
        image_dict = dict(
            primary =  cv2.resize(root[f'/observations/image/primary'][start_ts], eval(self.data_args.image_size_primary)),
            wrist_left = cv2.resize(root[f'/observations/image/wrist_left'][start_ts], eval(self.data_args.image_size_wrist)),
            wrist_right = cv2.resize(root[f'/observations/image/wrist_right'][start_ts], eval(self.data_args.image_size_wrist)),
        )
        # 加载推理信息
        reasoning = ""
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
            if 'language_instruction' in feats or len(feats)==0: data_dict['language_instruction'] = self.get_language_instruction() # 加载文本
            if 'state' in feats or len(feats)==0: data_dict['state'] = root[f'/observations/qpos'][()] # 加载状态
            if 'qpos' in feats or len(feats)==0: data_dict['qpos'] = root[f'/observations/qpos'][()]
            if 'qvel' in feats or len(feats)==0: data_dict['qvel'] = root[f'/observations/qvel'][()]
            if 'action' in feats or len(feats)==0: # 加载动作 
                data_dict['action'] = root[f'/action'][()] # 根据控制类型加载相应动作数据
                if self.ctrl_type=='delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type=='rel':
                    raise NotImplementedError("relative action was not implemented")
            image_dict = dict()
            if 'image_primary' in feats or 'image' in feats or len(feats)==0: # 加载图像
                img_bytes = root[f'/observations/image/primary'][()]
                image_dict.update(
                    dict(
                        primary =  np.stack([cv2.resize(img_byte, eval(self.data_args.image_size_primary)) for img_byte in img_bytes]),
                    )
                )
            if 'image_wrist' in feats or 'image' in feats or len(feats)==0:
                left_bytes = root[f'/observations/image/wrist_left'][()]
                image_dict.update(
                    dict(
                        wrist_left = np.stack([cv2.resize(lb, eval(self.data_args.image_size_wrist)) for lb in left_bytes]),
                    )
                )
                del left_bytes
                right_bytes = root[f'/observations/image/wrist_right'][()]
                image_dict.update(
                    dict(
                        wrist_right = np.stack([cv2.resize(rb ,eval(self.data_args.image_size_wrist)) for rb in right_bytes]),
                    )
                )
                del right_bytes
            if len(image_dict)>0: data_dict['image'] = image_dict
        return data_dict