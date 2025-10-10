import numpy as np
import torch

class OctoDataProcessor:
    def __init__(self, text_processor, use_wrist=False):
        self.text_processor = text_processor
        self.use_wrist = use_wrist
    
    def _np2pt(self, data, dtype=None):
        """Transform dictionary with numpy arrays to torch tensors.
        Trnsform images to channel-first format: NHWC -> NCHW, NTHWC -> NTCHW

        """
        if isinstance(data, dict):
            return {key: self._np2pt(val) for key, val in data.items()}
        t = torch.from_numpy(data)
        return t
    
    def __call__(self, sample):
        data_dict = {}
        # task
        task = {}
        task["language_instruction"] = self.text_processor.encode([s.decode("utf-8") for s in np.array([sample['raw_lang'].encode()])])
        task["pad_mask_dict"] = {"language_instruction": np.array([True])}
        task["language_instruction"]['input_ids'] = task["language_instruction"]['input_ids'][0]
        task["language_instruction"]['attention_mask'] = task["language_instruction"]['attention_mask'][0]
        data_dict['task'] = task 
        # observation
        obs = {}
        obs['proprio'] = sample['state'] # T*D
        obs['timestep'] = np.array(sample['timestamp'])
        obs['timestep_pad_mask'] = np.array([True])
        obs['task_completed'] = np.array([False for _ in range(sample['action'].shape[0])]) # (1 ,chunk_size)
        images = sample['image']
        obs['image_primary'] = sample['image'][0:1,:] # k c h w -> k h w c
        obs['pad_mask_dict'] = {
            'image_primary': np.array([True]),
            'proprio': np.array([True]),
            'timestep': np.array([True]),
        }
        if self.use_wrist:
            assert sample['image'].shape[0]>1
            obs['image_wrist'] = sample['image'][1:2,:]
            obs['pad_mask_dict']['image_wrist'] = np.array([True])
        data_dict['observation'] = obs
        # action
        data_dict['action'] = sample['action'][np.newaxis,:]
        data_dict['action_pad_mask'] = (1.-np.tile(sample['is_pad'][:, None], (1, data_dict['action'].shape[-1]))[np.newaxis,:]).bool()
        data_dict = self._np2pt(data_dict)
        return data_dict

        
        
        
        
        
