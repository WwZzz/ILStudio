from pprint import pprint
import torch.utils.data as tud
import torch
from huggingface_hub import HfApi
from typing import List
try:
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    import lerobot
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import warnings
from benchmark.utils import resize_with_pad
from data_utils.utils import ensure_uint8_image
from loguru import logger

class WrappedLerobotDataset(tud.Dataset):
    def __init__(self, 
            dataset_path_list: list, 
            camera_names: list=[], 
            root: str = None,
            chunk_size: int = 16,  
            ctrl_space: str = 'ee', 
            ctrl_type: str = 'delta',
            image_size: tuple = None,
            tolerance_s: float = 1e-4,
            state_key: str = 'observation.state',
            action_key: str = 'action',
            episode_filter: dict = None,  # Filter episodes by metadata fields, e.g., {"tasks": ["task1"], "episode_index": [0, 1, 2]}
            *args, 
            **kwargs,
            ):
        super().__init__()
        self.chunk_size = chunk_size
        self.root = root
        self.state_key = state_key
        self.action_key = action_key
        self.episode_filter = episode_filter
        datasets = []
        data_metas = []
        dataset_dirs = []
        num_episodes = []
        num_frames = []
        all_camera_keys = dict()
        for data_path in dataset_path_list:
            ds_meta = LeRobotDatasetMetadata(data_path, root=self.root)
            
            # Filter episodes by provided filter conditions
            episodes_to_load = None
            if episode_filter is not None and len(episode_filter) > 0:
                episodes_to_load = self._filter_episodes(ds_meta, episode_filter)
                if episodes_to_load is not None and len(episodes_to_load) == 0:
                    warnings.warn(f"No episodes found matching filter {episode_filter} in dataset {data_path}")
                    continue
            
            delta_timestamps = {self.action_key: [t / ds_meta.fps for t in range(chunk_size)]}
            dataset = LeRobotDataset(
                data_path, 
                root=self.root, 
                delta_timestamps=delta_timestamps, 
                tolerance_s=tolerance_s,
                episodes=episodes_to_load
            )
            
            # Log filtering info and calculate actual frames count
            if episodes_to_load is not None:
                logger.info(f"Selected {len(episodes_to_load)} episodes from {data_path}")
                logger.info(f"Episode indices: {episodes_to_load[:10]}{'...' if len(episodes_to_load) > 10 else ''}")
                
                # Calculate actual frame count for filtered episodes
                # LeRobot doesn't actually reduce hf_dataset size, so we need to calculate manually
                actual_frames = sum(ds_meta.episodes[ep_idx]['length'] for ep_idx in episodes_to_load)
                # logger.info(f"Actual frames for selected episodes: {actual_frames}")
            else:
                actual_frames = dataset.num_frames
            
            data_metas.append(ds_meta)
            datasets.append(dataset)
            dataset_dirs.append(str(dataset.root))
            # Use actual dataset size after filtering
            num_episodes.append(dataset.num_episodes)
            num_frames.append(actual_frames)  # Use calculated frames, not dataset.num_frames
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
        
    def _filter_episodes(self, ds_meta: LeRobotDatasetMetadata, episode_filter: dict) -> list:
        """
        Filter episodes based on metadata fields. This is a universal filtering method
        that works across all LeRobot datasets.
        
        Args:
            ds_meta: LeRobotDatasetMetadata object
            episode_filter: Dictionary containing filter conditions. Supported keys:
                - "episode_index": list of episode indices to include (directly)
                - "tasks": list of task names (if dataset has tasks)
                - "task_index": list of task indices (if dataset has tasks)
                - Any other field in episodes metadata can be used
                
        Returns:
            List of episode indices that match the filter conditions.
            Returns None if filtering is not possible.
            
        Examples:
            # Direct episode indices
            {"episode_index": [0, 1, 2, 5, 10]}
            
            # Filter by tasks (if dataset has tasks)
            {"tasks": ["reach-v2", "push-v2"]}
            
            # Filter by task index (if dataset has tasks)
            {"task_index": [0, 1, 2]}
            
            # Combine multiple filters (AND logic)
            {"tasks": ["reach-v2"], "episode_index": [0, 1, 2]}
        """
        if ds_meta.episodes is None:
            warnings.warn(f"Dataset metadata does not contain episodes information")
            return None
        
        # If episode_index is directly provided, use it
        if "episode_index" in episode_filter:
            episode_indices = episode_filter["episode_index"]
            if isinstance(episode_indices, (list, tuple)):
                # Validate indices
                valid_indices = [idx for idx in episode_indices if 0 <= idx < len(ds_meta.episodes)]
                if len(valid_indices) != len(episode_indices):
                    warnings.warn(f"Some episode indices are out of range. Using {len(valid_indices)}/{len(episode_indices)} valid indices.")
                return valid_indices
            else:
                warnings.warn(f"episode_index should be a list, got {type(episode_indices)}")
                return None
        
        # Start with all episodes
        selected_episodes = set(range(len(ds_meta.episodes)))
        
        # Apply each filter condition
        for filter_key, filter_values in episode_filter.items():
            if filter_key == "tasks":
                # Filter by task names
                if ds_meta.tasks is None:
                    warnings.warn(f"Dataset does not have tasks metadata, cannot filter by 'tasks'")
                    return None
                
                task_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_tasks = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    episode_tasks = ds_meta.episodes[episode_idx].get('tasks', None)
                    if episode_tasks is None:
                        continue
                    
                    # episode_tasks could be a list or single value
                    if isinstance(episode_tasks, (list, tuple)):
                        if any(task in task_set for task in episode_tasks):
                            episodes_for_tasks.add(episode_idx)
                    else:
                        if episode_tasks in task_set:
                            episodes_for_tasks.add(episode_idx)
                
                selected_episodes &= episodes_for_tasks
                
            elif filter_key == "task_index":
                # Filter by task indices
                if ds_meta.tasks is None:
                    warnings.warn(f"Dataset does not have tasks metadata, cannot filter by 'task_index'")
                    return None
                
                task_idx_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_task_idx = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    # Get task names for this episode
                    episode_tasks = ds_meta.episodes[episode_idx].get('tasks', None)
                    if episode_tasks is None:
                        continue
                    
                    # Convert task names to indices
                    if isinstance(episode_tasks, (list, tuple)):
                        episode_task_indices = [ds_meta.get_task_index(task) for task in episode_tasks]
                    else:
                        episode_task_indices = [ds_meta.get_task_index(episode_tasks)]
                    
                    # Check if any task index matches
                    if any(idx in task_idx_set for idx in episode_task_indices if idx is not None):
                        episodes_for_task_idx.add(episode_idx)
                
                selected_episodes &= episodes_for_task_idx
                
            else:
                # Generic filter by any episode metadata field
                filter_value_set = set(filter_values) if isinstance(filter_values, (list, tuple)) else {filter_values}
                episodes_for_field = set()
                
                for episode_idx in range(len(ds_meta.episodes)):
                    episode_value = ds_meta.episodes[episode_idx].get(filter_key, None)
                    if episode_value is None:
                        continue
                    
                    # Handle list values
                    if isinstance(episode_value, (list, tuple)):
                        if any(val in filter_value_set for val in episode_value):
                            episodes_for_field.add(episode_idx)
                    else:
                        if episode_value in filter_value_set:
                            episodes_for_field.add(episode_idx)
                
                if len(episodes_for_field) == 0:
                    warnings.warn(f"No episodes found with {filter_key} in {filter_values}")
                
                selected_episodes &= episodes_for_field
        
        return sorted(list(selected_episodes))
        
    def initialize(self):
        self.episode_len = self.get_episode_len() 
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(self.episode_len)
        
        # Build index mapping table for fast lookup (critical for performance!)
        self._build_index_mapping()
        
        # Log dataset info
        logger.info(f"Dataset initialized: {self.total_episodes} episodes, {self.total_frames} frames")
        # logger.info(f"Episode lengths: min={min(self.episode_len)}, max={max(self.episode_len)}, mean={np.mean(self.episode_len):.1f}")
        return
    
    def _build_index_mapping(self):
        """
        Pre-compute index mapping for filtered datasets to avoid runtime loops.
        This is CRITICAL for performance with multi-worker dataloaders!
        """
        self.index_to_episode_map = []  # [(dataset_idx, episode_idx, frame_offset_in_episode, actual_hf_index), ...]
        
        for dataset_idx, dataset in enumerate(self.datasets):
            if hasattr(dataset, 'episodes') and dataset.episodes is not None:
                # Filtered dataset - need to build mapping
                ds_meta = self.dataset_metas[dataset_idx]
                for ep_idx in dataset.episodes:
                    ep_len = ds_meta.episodes[ep_idx]['length']
                    ep_start_in_hf = ds_meta.episodes[ep_idx]['dataset_from_index']
                    
                    for frame_offset in range(ep_len):
                        actual_hf_idx = ep_start_in_hf + frame_offset
                        self.index_to_episode_map.append((dataset_idx, ep_idx, frame_offset, actual_hf_idx))
            else:
                # No filtering - direct 1:1 mapping
                total_frames = self.per_dataset_num_frames[dataset_idx]
                for frame_idx in range(total_frames):
                    self.index_to_episode_map.append((dataset_idx, -1, -1, frame_idx))
        
        # logger.info(f"Built index mapping table with {len(self.index_to_episode_map)} entries")
        
    def _load_file_into_memory(self, *args, **kwargs):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def _load_all_episodes_into_memory(self):
        warnings.warn("Cannot load LerobotDataset into memory")
        return
    
    def get_episode_len(self):
        """
        Get the length of each episode in the filtered dataset.
        Must use the filtered dataset, not the original metadata.
        """
        episode_lens = []
        for dataset in self.datasets:
            # If dataset has episodes attribute (filtered list), use metadata for those episodes
            # This is much faster than iterating through hf_dataset
            if hasattr(dataset, 'episodes') and dataset.episodes is not None:
                ds_meta = dataset.meta
                for ep_idx in dataset.episodes:
                    episode_lens.append(ds_meta.episodes[ep_idx]['length'])
            elif dataset.hf_dataset is not None:
                # Fallback: use numpy for efficient counting
                # Convert to numpy array for fast operations
                episode_indices_array = np.array(dataset.hf_dataset['episode_index'])
                unique_episodes = np.unique(episode_indices_array)
                
                # Use numpy for efficient counting
                for ep_idx in unique_episodes:
                    ep_length = np.sum(episode_indices_array == ep_idx)
                    episode_lens.append(int(ep_length))
        return episode_lens
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        # Use index mapping table length for accuracy
        if hasattr(self, 'index_to_episode_map'):
            return len(self.index_to_episode_map)
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
        """
        Locate which dataset and frame index for a given wrapped index.
        Uses pre-computed mapping table for O(1) lookup - FAST!
        """
        assert index < len(self.index_to_episode_map), f"Index {index} out of range {len(self.index_to_episode_map)}"
        
        # O(1) lookup from pre-computed table!
        dataset_idx, ep_idx, frame_offset, actual_hf_idx = self.index_to_episode_map[index]
        
        return int(dataset_idx), int(actual_hf_idx)
    
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
        ori_k = {}
        if 'state' in keyname:
            preserved_keys.append(self.state_key)
            ori_k[self.state_key] = 'state'
        if 'action' in keyname:
            preserved_keys.append(self.action_key)
            ori_k[self.action_key] = 'action'
        if 'image' in keyname or 'images' in keyname:
            preserved_keys.extend(ds_meta.camera_keys)
            for i,k in enumerate(ds_meta.camera_keys):
                ori_k[k] = f'images_{i}'
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
        res_dict = {ori_k[k]: np.stack([efeat[k] for efeat in  extracted_feats]) if isinstance(extracted_feats[0][k], np.ndarray) else [efeat[k] for efeat in  extracted_feats] for k in preserved_keys}
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
        action = sample[self.action_key]
        state = sample[self.state_key]
        timestamp = sample['frame_index'].item()
        is_pad = sample['action_is_pad']
        # process image
        cam_keys = self.datasets[dataset_idx].meta.camera_keys if len(self.camera_names)==0 else self.camera_names
        if self.image_size is not None:
            images = torch.cat([resize_with_pad(sample[cam_key].unsqueeze(0), height=self.image_size[1], width=self.image_size[0]) for cam_key in cam_keys], dim=0)
        else:
            images = torch.stack([sample[cam_key] for cam_key in cam_keys])
        
        # Safety check: ensure images are uint8 with values in [0, 255]
        images = ensure_uint8_image(images)
        
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

    def get_dataset_statistics(self):
        state_stats = self.dataset_metas[0].stats[self.state_key]
        action_stats = self.dataset_metas[0].stats[self.action_key]
        if 'q01' not in state_stats:
            state_stats['q01'] = state_stats['min']
            state_stats['q99'] = state_stats['max']
        if 'q01' not in action_stats:
            action_stats['q01'] = action_stats['min']
            action_stats['q99'] = action_stats['max']
        stats = {
            'state': state_stats,
            'action': action_stats,
            'num_episodes': self.total_episodes,
            'num_transitions': self.total_frames,
        }
        return stats
    
        
if __name__=='__main__':
    ##################################### Liberos #################################
    dataset = WrappedLerobotDataset(
            ["HuggingFaceVLA/libero"], 
            tolerance_s=10.0,
        )
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total frames: {dataset.total_frames}")
    dataset = WrappedLerobotDataset(
            ["HuggingFaceVLA/libero"], 
            tolerance_s=10.0,
            episode_filter={"task_index": [24]}
        )
    print(f"Task0 Total episodes: {dataset.total_episodes}")
    print(f"Task0 Total frames: {dataset.total_frames}")
    ##################################### MetaWorld #################################
    # # Example 1: Load dataset without filtering
    # print("=" * 60)
    # print("Example 1: Load entire dataset (no filtering)")
    # print("=" * 60)
    # dataset = WrappedLerobotDataset(["lerobot/metaworld_mt50", ], tolerance_s=10.0)
    # print(f"Total episodes: {dataset.total_episodes}")
    # print(f"Total frames: {dataset.total_frames}")
    
    # # Print available tasks if exists
    # print("\nDataset info:")
    # if dataset.dataset_metas[0].tasks is not None:
    #     print(f"Has tasks: Yes ({len(dataset.dataset_metas[0].tasks)} tasks)")
    #     print("Available tasks:", list(dataset.dataset_metas[0].tasks.index[:5]))
    # else:
    #     print("Has tasks: No")
    
    # # Example 2: Filter by episode_index directly (works for all datasets)
    # print("\n" + "=" * 60)
    # print("Example 2: Filter by episode_index (universal method)")
    # print("=" * 60)
    # print("Loading episodes [0, 1, 2, 5, 10]")
    # dataset_by_index = WrappedLerobotDataset(
    #     ["lerobot/metaworld_mt50"], 
    #     tolerance_s=10.0,
    #     episode_filter={"episode_index": [0, 1, 2, 5, 10]}
    # )
    # print(f"Filtered episodes: {dataset_by_index.total_episodes}")
    # print(f"Filtered frames: {dataset_by_index.total_frames}")
    # print(f"Dataset length (__len__): {len(dataset_by_index)}")
    # print(f"Episode lengths: {dataset_by_index.episode_len[:10]}")  # Show first 10
    
    # # Example 3: Filter by tasks (only if dataset has tasks)
    # print("\n" + "=" * 60)
    # print("Example 3: Filter by tasks (if available)")
    # print("=" * 60)
    
    # if dataset.dataset_metas[0].tasks is not None and len(dataset.dataset_metas[0].tasks) > 0:
    #     example_tasks = list(dataset.dataset_metas[0].tasks.index[:2])
    #     print(f"Filtering by tasks: {example_tasks}")
        
    #     dataset_by_task = WrappedLerobotDataset(
    #         ["lerobot/metaworld_mt50"], 
    #         tolerance_s=10.0,
    #         episode_filter={"tasks": example_tasks}
    #     )
    #     print(f"Filtered episodes: {dataset_by_task.total_episodes}")
    #     print(f"Filtered frames: {dataset_by_task.total_frames}")
    #     print(f"Dataset length (__len__): {len(dataset_by_task)}")
    #     print(f"First 5 episode lengths: {dataset_by_task.episode_len[:5]}")
    # else:
    #     print("No tasks found in dataset metadata")
    

    # print(f"Filtered frames: {dataset_by_index.total_frames}")
    # print(f"Dataset length (__len__): {len(dataset_by_index)}")
    # print(f"Episode lengths: {dataset_by_index.episode_len[:10]}")  # Show first 10
    
    # # Example 3: Filter by tasks (only if dataset has tasks)
    # print("\n" + "=" * 60)
    # print("Example 3: Filter by tasks (if available)")
    # print("=" * 60)
    
    # if dataset.dataset_metas[0].tasks is not None and len(dataset.dataset_metas[0].tasks) > 0:
    #     example_tasks = list(dataset.dataset_metas[0].tasks.index[:2])
    #     print(f"Filtering by tasks: {example_tasks}")
        
    #     dataset_by_task = WrappedLerobotDataset(
    #         ["lerobot/metaworld_mt50"], 
    #         tolerance_s=10.0,
    #         episode_filter={"tasks": example_tasks}
    #     )
    #     print(f"Filtered episodes: {dataset_by_task.total_episodes}")
    #     print(f"Filtered frames: {dataset_by_task.total_frames}")
    #     print(f"Dataset length (__len__): {len(dataset_by_task)}")
    #     print(f"First 5 episode lengths: {dataset_by_task.episode_len[:5]}")
    # else:
    #     print("No tasks found in dataset metadata")
    
