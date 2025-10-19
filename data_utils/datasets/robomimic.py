"""
RobomimicDataset implementation.

This module contains the RobomimicDataset class for loading RoboMimic datasets.
"""

import numpy as np
import os
from collections import OrderedDict
from data_utils.rotate import quat2axisangle
from .base import EpisodicDataset


class RobomimicDataset(EpisodicDataset):
    """
    Dataset class for RoboMimic data.
    
    This class handles loading and processing of RoboMimic datasets,
    which include various manipulation tasks from the RoboMimic benchmark.
    """
    
    def initialize(self):
        """Initialize the RoboMimic dataset with sequence datasets."""
        from robomimic.utils.dataset import SequenceDataset
        self._datasets = [SequenceDataset(**self.create_config(di)) for di in self.dataset_path_list if 'image' in di]
        self._languages = [self.get_raw_lang(di) for di in self.dataset_path_list if 'image' in di]
        self._dataset_dir = os.path.dirname(self.dataset_path_list[0])
        self.episode_ids = np.arange(sum(d.n_demos for d in self._datasets))
        self.dataset_path_list = sum([[f"{idx}:{ep}" for ep in di.demos] for idx, di in enumerate(self._datasets)], [])
        self.episode_len = self.get_episode_len()  # Get length of each episode
        self.cumulative_len = np.cumsum(self.episode_len)  # Compute cumulative lengths
        self.max_episode_len = max(self.episode_len)  # Get maximum episode length

    def get_dataset_dir(self):
        """Get the dataset directory path."""
        return self._dataset_dir
    
    def get_raw_lang(self, data_path):
        """
        Get raw language instruction based on the data path.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            String containing the language instruction
        """
        # Initialize raw language
        from benchmark.robomimic import ALL_ENV_LANGUAGES
        if 'can' in data_path: 
            return ALL_ENV_LANGUAGES['PickPlaceCan']
        elif 'lift' in data_path: 
            return ALL_ENV_LANGUAGES['Lift']
        elif 'tool_hang' in data_path: 
            return ALL_ENV_LANGUAGES['ToolHang']
        elif 'square' in data_path: 
            return ALL_ENV_LANGUAGES['NutAssemblySquare']
        elif 'transport' in data_path: 
            return ALL_ENV_LANGUAGES['TwoArmTransport']
        else:
            raise KeyError("Unknown language")
        
    def create_config(self, data_path, filter_by_attribute='train', seq_length=1):
        """
        Create configuration for the RoboMimic sequence dataset.
        
        Args:
            data_path: Path to the data file
            filter_by_attribute: Attribute to filter by
            seq_length: Sequence length
            
        Returns:
            Dictionary containing the configuration
        """
        obs_key_shapes = OrderedDict([
            ('agentview_image', [3, 84, 84]), 
            ('robot0_eef_pos', [3]), 
            ('robot0_eef_quat', [4]), 
            ('robot0_eye_in_hand_image', [3, 84, 84]), 
            ('robot0_gripper_qpos', [2])
        ])
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
            'hdf5_cache_mode': 'all' if self.preload_data else 'low_dim',
            'hdf5_use_swmr': True,
            'hdf5_normalize_obs': False,
            'filter_by_attribute': filter_by_attribute,
        }
    
    def get_episode_len(self):
        """
        Get the length of each episode in the dataset.
        
        Returns:
            List of episode lengths
        """
        all_ep_lengths = []
        for dataset_path in self.dataset_path_list:
            idx, ep = dataset_path.split(':')
            ep_length = self._datasets[eval(idx)]._demo_id_to_demo_length[ep]
            all_ep_lengths.append(ep_length)
        return all_ep_lengths
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        demo_start_index = dataset._demo_id_to_start_indices[ep]  # Demo start global index
        global_index = demo_start_index + start_ts
        data = dataset[global_index]
        # Load language 
        raw_lang = self._languages[eval(dataset_idx)]
        # Load state
        state_euler = quat2axisangle(data['obs']['robot0_eef_quat'])
        state_xyz = data['obs']['robot0_eef_pos']
        gpos = data['obs']['robot0_gripper_qpos']
        state_gripper = (gpos[:, 0] - gpos[:, 1])[:, np.newaxis]
        state = np.concatenate([state_xyz, state_euler, state_gripper], axis=1)[0]
        # Load action
        if dataset.hdf5_cache_mode == 'all':
            demo_length = dataset._demo_id_to_demo_length[ep]
            chunk_size = min(self.chunk_size, demo_length - start_ts)
            action = np.concatenate([data['actions']] + [dataset[i]['actions'] for i in range(global_index + 1, global_index + chunk_size)], axis=0)
        else:
            action = data['actions'] if self.chunk_size == 1 else dataset.get_dataset_for_ep(ep=ep, key="actions")[start_ts:start_ts + self.chunk_size]
        action[:, :6] = action[:, :6] * np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        action[:, -1] = 0.5 * (1 - action[:, -1])
        if self.ctrl_type == 'abs':
            if dataset.hdf5_cache_mode == 'all':
                next_data = [data] + [dataset[i] for i in range(global_index + 1, global_index + chunk_size)]
                xyzs = np.concatenate([di['obs']['robot0_eef_pos'] for di in next_data], axis=0)
                eulers = quat2axisangle(np.concatenate([di['obs']['robot0_eef_quat'] for di in next_data], axis=0))
                states = np.concatenate([xyzs, eulers], axis=1)
            else:
                xyzs = dataset.get_dataset_for_ep(ep=ep, key="obs/robot0_eef_pos")[start_ts:start_ts + self.chunk_size]
                eulers = quat2axisangle(dataset.get_dataset_for_ep(ep=ep, key="obs/robot0_eef_quat")[start_ts:start_ts + self.chunk_size]) 
                states = np.concatenate([xyzs, eulers], axis=1)
            action[:, :6] = action[:, :6] + states
        # Load image
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
        """
        Load all steps data from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        dataset_idx, ep = dataset_path.split(':')
        dataset = self._datasets[eval(dataset_idx)]
        if dataset.hdf5_cache_mode == 'all':
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
        if isinstance(feats, str): 
            feats = [feats]
        if 'language_instruction' in feats or len(feats) == 0: 
            data_dict['language_instruction'] = self._languages[dataset_idx]
        if 'state' in feats or len(feats) == 0: 
            state_euler = quat2axisangle(trajectory_data['obs']['robot0_eef_quat'])
            state_xyz = trajectory_data['obs']['robot0_eef_pos']
            gpos = trajectory_data['obs']['robot0_gripper_qpos']
            state_gripper = (gpos[:, 0] - gpos[:, 1])[:, np.newaxis]
            state = np.concatenate([state_xyz, state_euler, state_gripper], axis=1)
            data_dict['state'] = state
        if 'action' in feats or len(feats) == 0:  # Load action
            action = trajectory_data['actions']
            action[:, :6] = action[:, :6] * np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
            action[:, -1] = 0.5 * (1 - action[:, -1])
            data_dict['action'] = action
        if 'image' in feats or len(feats) == 0:  # Load images
            image_dict = dict(
                primary=trajectory_data['obs']['agentview_image'],
                wrist=trajectory_data['obs']['robot0_eye_in_hand_image']
            )
            data_dict['image'] = image_dict
        return data_dict
