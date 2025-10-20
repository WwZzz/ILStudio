from pprint import pprint
import torch.utils.data as tud
import torch
from huggingface_hub import HfApi
from typing import List
try:
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    lerobot = None
import numpy as np
import warnings
from benchmark.utils import resize_with_pad

class WrappedLerobotDataset(tud.Dataset):
    def __init__(self, 
            dataset_path_list: list, 
            camera_names: list=[], 
            chunk_size: int = 16,  
            ctrl_space: str = 'ee', 
            ctrl_type: str = 'delta',
            image_size: tuple = None,
            *args, 
            **kwargs,
            ):
        super().__init__()
        self.chunk_size = chunk_size
        datasets = []
        data_metas = []
        dataset_dirs = []
        num_episodes = []
        num_frames = []
        all_camera_keys = dict()
        for data_path in dataset_path_list:
            ds_meta = LeRobotDatasetMetadata(data_path)
            delta_timestamps = {'action': [t / ds_meta.fps for t in range(chunk_size)]}
            dataset = LeRobotDataset(data_path, delta_timestamps=delta_timestamps)
            data_metas.append(ds_meta)
            datasets.append(dataset)
            dataset_dirs.append(str(dataset.root))
            num_episodes.append(ds_meta.total_episodes)
            num_frames.append(ds_meta.total_frames)
            all_camera_keys[data_path] = ds_meta.camera_keys
        self.dataset_path_list = dataset_path_list
        self.datasets = datasets
        self.dataset_metas= data_metas
        self.dataset_dirs = dataset_dirs
        self.per_dataset_num_episodes = num_episodes
        self.per_dataset_num_frames = num_frames
        self.cumulative_num_episodes = np.cumsum(self.per_dataset_num_episodes)
        self.cumulative_num_frames = np.cumsum(self.per_dataset_num_frames)
        self.per_dataset_episode_start = self.cumulative_num_episodes - np.array(self.per_dataset_num_episodes)
        self.per_dataset_frame_start = self.cumulative_num_frames - np.array(self.per_dataset_num_frames)
        self.total_frames = sum(self.per_dataset_num_frames)
        self.total_episodes = sum(num_episodes)
        self.camera_names = camera_names if isinstance(camera_names, list) else [camera_names]
        self.episode_ids = np.arange(sum(self.per_dataset_num_episodes))
        self.image_size = image_size
        self.ctrl_space = ctrl_space  # ['ee', 'joint', 'other']
        self.ctrl_type = ctrl_type  # ['abs', 'rel', 'delta']
        self.freq = self.dataset_metas[0].fps
        self.max_workers = 8
        self.initialize()
        
    def initialize(self):
        self.episode_len = self.get_episode_len() 
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(self.episode_len)
        return
        
    def _load_file_into_memory(self, *args, **kwargs):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def _load_all_episodes_into_memory(self):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def get_episode_len(self):
        episode_lens = []
        for ds_meta in self.dataset_metas:
            episode_lens.extend([e['length'] for e in ds_meta.episodes.values()])
        return episode_lens
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_frames
        
    @property
    def num_episodes(self):
        return self.total_episodes

    def get_dataset_dir(self):
        """Get the dataset directory path."""
        return self.dataset_dirs[0]
       
    def get_freq(self):
        """Get the dataset frequency."""
        return self.freq
    
    def _locate_dataset_for_transition(self, index):
        assert index < self.cumulative_num_frames[-1]
        dataset_idx = np.argmax(self.cumulative_num_frames>index)
        start_ts = index - self.per_dataset_frame_start[dataset_idx]
        return int(dataset_idx), int(start_ts)
    
    def _locate_transition(self, index):
        """
        Convert sample index to episode index and internal timestep.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (episode_id, start_ts)
        """
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts
    
    def extract_from_episode(self, episode_idx, keyname=[]):
        dataset_idx = np.argmax(self.cumulative_num_episodes > episode_idx)
        inner_episode_idx = episode_idx - self.per_dataset_episode_start[dataset_idx]
        ds_meta = self.dataset_metas[dataset_idx]
        all_features = ds_meta.features
        preserved_keys = []
        if 'state' in keyname:
            preserved_keys.append('observation.state')
        if 'action' in keyname:
            preserved_keys.append('action')
        if 'image' in keyname or 'images' in keyname:
            preserved_keys.extend(ds_meta.camera_keys)
            ignore_image = False
        else:
            ignore_image = all([ckey not in keyname for ckey in ds_meta.camera_keys])
        ignore_keys = [feat for feat in all_features if feat not in preserved_keys and feat not in keyname]
        subdata = LeRobotDataset(
            self.dataset_path_list[dataset_idx], 
            episodes=[inner_episode_idx],
        )
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='video': subdata.meta.info['features'][k]['dtype'] = 'hidden'
        extracted_feats = [{k:s[k].numpy() for k in preserved_keys} for s in subdata]
        if ignore_image:
            for k,v in subdata.meta.features.items():
                if v['dtype']=='hidden': subdata.meta.info['features'][k]['dtype'] = 'video'
        res_dict = {k: np.stack([efeat[k] for efeat in  extracted_feats]) if isinstance(extracted_feats[0][k], np.ndarray) else [efeat[k] for efeat in  extracted_feats] for k in preserved_keys}
        return res_dict
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        # find dataset_id by index: start_index for the target dataset by the num_frames of each dataset
        # find sample_id in dataset: index-start_index
        
        dataset_idx, start_ts = self._locate_dataset_for_transition(index)
        sample = self.datasets[dataset_idx][start_ts]
        data_dict = {}
        episode_id = self.per_dataset_episode_start[dataset_idx] + sample['episode_index'].item()
        raw_lang = sample['task']
        action = sample['action']
        state = sample['observation.state']
        timestamp = sample['frame_index'].item()
        is_pad = sample['action_is_pad']
        # process image
        cam_keys = self.datasets[dataset_idx].meta.camera_keys if len(self.camera_names)==0 else self.camera_names
        if self.image_size is not None:
            images = torch.cat([resize_with_pad(sample[cam_key].unsqueeze(0), height=self.image_size[1], width=self.image_size[0]) for cam_key in cam_keys], dim=0)
        else:
            images = torch.stack([sample[cam_key] for cam_key in cam_keys])
        data_dict = {
            'image': images,
            'state': state,
            'action': action,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': {},
            'timestamp': timestamp,  
            'episode_id': episode_id,
        }  
        return data_dict
    
        
# if __name__=='__main__':
#     dataset = WrappedLerobotDataset(["lerobot/aloha_mobile_cabinet", "lerobot/aloha_mobile_cabinet"])
#     d = dataset.extract_from_episode(86, ['state', 'action'])
#     print('ok')