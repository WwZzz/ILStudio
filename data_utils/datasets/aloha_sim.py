"""
AlohaSimDataset implementation.

This module contains the AlohaSimDataset class for loading ALOHA simulation data.
"""

import numpy as np
import cv2
import h5py
import os
from .base import EpisodicDataset


class AlohaSimDataset(EpisodicDataset):
    """
    Dataset class for ALOHA simulation data.
    
    This class handles loading and processing of ALOHA simulation datasets,
    including transfer and insertion tasks.
    """
    
    def get_language_instruction(self):
        """
        Get language instruction based on the task name.
        
        Returns:
            String containing the task instruction
        """
        task_name = os.path.split(self.get_dataset_dir())[-1]
        if 'transfer' in task_name:
            return 'Transfer the red cube from the right arm to the left arm.'
        elif 'insert' in task_name:
            return 'Insert the red peg into the blue socket.'
        else:
            raise ValueError("Unknown task")
        
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        """
        Load one-step data at start_ts from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            start_ts: Starting timestep
            
        Returns:
            Dictionary containing the loaded data
        """
        root = self.loaded_data[dataset_path] if self.loaded_data is not None else h5py.File(dataset_path, 'r')
        # Load text
        raw_lang = self.get_language_instruction()
        # Load action & state
        action = root[f'/action'][start_ts:start_ts+self.chunk_size]
        # Load corresponding action data based on control type
        if self.ctrl_type == 'abs':
            state = root[f'/observations/qpos'][start_ts]
        elif self.ctrl_type == 'delta':
            states = root[f'/observations/qpos'][start_ts:start_ts+self.chunk_size]
            action = action - states
            state = states[0]
        elif self.ctrl_type == 'rel':
            raise NotImplementedError("relative action was not implemented")
        # Load images
        image_dict = dict(primary=cv2.resize(root['/observations/images/top'][start_ts], self.data_args.image_size))
        if 'left_wrist' in self.data_args.camera_names:
            if '/observations/images/left_wrist' in root:
                image_dict.update(
                    dict(wrist_left=cv2.resize(root['/observations/images/left_wrist'][start_ts], self.data_args.image_size))
                )
        if 'right_wrist' in self.data_args.camera_names:
            if '/observations/images/right_wrist' in root:
                image_dict.update(
                    dict(wrist_right=cv2.resize(root['/observations/images/right_wrist'][start_ts], self.data_args.image_size))
                )
        # Load reasoning information
        reasoning = ""
        if self.loaded_data is None: 
            root.close()
        return {
            'action': action,
            'image': image_dict,
            'state': state,
            'language_instruction': raw_lang,
            'reasoning': reasoning,
        }
        
    def load_feat_from_episode(self, dataset_path, feats=[]):
        """
        Load all steps data from the episode specified by dataset_path.
        
        Args:
            dataset_path: Path to the dataset file
            feats: List of features to load
            
        Returns:
            Dictionary containing the loaded data
        """
        data_dict = {}
        if isinstance(feats, str): 
            feats = [feats]
        with h5py.File(dataset_path, 'r') as root:
            if 'language_instruction' in feats or len(feats) == 0: 
                data_dict['language_instruction'] = self.get_language_instruction()  # Load text
            if 'state' in feats or len(feats) == 0: 
                data_dict['state'] = root[f'/observations/qpos'][()]  # Load state
            if 'action' in feats or len(feats) == 0:  # Load action
                data_dict['action'] = root[f'/action'][()]  # Load corresponding action data based on control type
                if self.ctrl_type == 'delta': 
                    data_dict['action'] = data_dict['action'] - data_dict.get('state', root[f'/observations/qpos'][()])
                elif self.ctrl_type == 'rel':
                    raise NotImplementedError("relative action was not implemented")
            if 'image' in feats or len(feats) == 0:  # Load images
                all_images = root[f'/observations/images/top'][()]
                image_dict = dict(primary=[cv2.resize(all_images[i], self.data_args.image_size) for i in range(all_images.shape[0])])
                data_dict['image'] = image_dict
            if 'image' in feats or 'image_wrist' in feats or len(feats) == 0:
                all_left_images = root[f'/observations/images/left_wrist'][()]
                left_dict = dict(wrist_left=[cv2.resize(all_left_images[i], self.data_args.image_size) for i in range(all_left_images.shape[0])])
                all_right_images = root[f'/observations/images/right_wrist'][()]
                right_dict = dict(wrist_right=[cv2.resize(all_right_images[i], self.data_args.image_size) for i in range(all_right_images.shape[0])])
                data_dict['image'].update(left_dict)
                data_dict['image'].update(right_dict)
        return data_dict
