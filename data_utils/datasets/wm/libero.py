import numpy as np
import torch
import h5py
import cv2

from data_utils.datasets.libero_h5 import LiberoHDF5
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image
from typing import List


class LiberoVideoDataset(LiberoHDF5):

    def __init__(self, root: str = "", split: List[str] = ['object'],
                 camera_names: List[str] = ['image_primary'],
                 chunk_size: int = 16,
                 ctrl_space: str = 'ee', ctrl_type: str = 'delta',
                 image_size: tuple = (256, 256), preload_data: bool = False,
                 horizon: int = 16, frame_skip: int = 1, input_steps: int = 1):
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.input_steps = input_steps
        super().__init__(root, split, camera_names, chunk_size,
                         ctrl_space, ctrl_type, image_size, preload_data)

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        h5path, demo = dataset_path.split(':')
        episode_len = self.episode_len[episode_id]

        frame_indices_unclipped = np.arange(self.horizon) * self.frame_skip + start_ts
        valid_mask = frame_indices_unclipped < episode_len
        is_pad_video = (~valid_mask).tolist()
        frame_indices = np.clip(frame_indices_unclipped, 0, episode_len - 1)

        root = self.loaded_data[h5path] if self.loaded_data is not None else h5py.File(h5path, 'r')
        prefix = f'/data/{demo}/'

        try:
            agentview = np.array(root[prefix + 'obs/agentview_rgb'][frame_indices])
            wrist = None
            if 'image_wrist' in self.camera_names:
                wrist = np.array(root[prefix + 'obs/eye_in_hand_rgb'][frame_indices])

            state = np.concatenate([
                root[prefix + 'obs/ee_states'][start_ts],
                root[prefix + 'obs/gripper_states'][start_ts],
            ], axis=0)

            action_end = min(start_ts + self.chunk_size, root[prefix + 'actions'].shape[0])
            action = root[prefix + 'actions'][start_ts:action_end]
        finally:
            if self.loaded_data is None and isinstance(root, h5py.File):
                root.close()

        if agentview.shape[1:3] != tuple(self.image_size):
            agentview = np.stack([cv2.resize(f, self.image_size) for f in agentview])
        cam_list = [agentview]
        if wrist is not None:
            if wrist.shape[1:3] != tuple(self.image_size):
                wrist = np.stack([cv2.resize(f, self.image_size) for f in wrist])
            cam_list.append(wrist)

        # (T, K, H, W, C)
        images = np.stack(cam_list, axis=1)
        images = ensure_uint8_image(images)
        num_views = images.shape[1]
        images = torch.from_numpy(images)
        # (T, K, H, W, C) -> (T, K, C, H, W) -> (T*K, C, H, W)
        images = images.permute(0, 1, 4, 2, 3).reshape(-1, images.shape[4], images.shape[2], images.shape[3])

        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded_action[:action.shape[0]] = action
        action_is_pad = np.zeros(self.chunk_size, dtype=bool)
        action_is_pad[action.shape[0]:] = True

        raw_lang = self._languages[h5path]
        freq = self.freq if self.freq > 0 else 10.0
        video_meta = build_video_meta(num_views, self.horizon, is_pad_video,
                                      self.input_steps, freq)

        return {
            'image': images,
            'state': torch.from_numpy(state).float(),
            'action': torch.from_numpy(padded_action).float(),
            'is_pad': torch.from_numpy(action_is_pad).bool(),
            'raw_lang': raw_lang,
            'reasoning': {"video": video_meta},
            'timestamp': start_ts,
            'episode_id': episode_id,
            '__index__': index,
        }
