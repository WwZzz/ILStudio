"""
ALOHA Sim Video Dataset – batch HDF5 reads for T consecutive frames.

HDF5 layout per file::

    /observations/images/top         → (T, H, W, C) uint8
    /observations/images/left_wrist  → (T, H, W, C) uint8  (optional)
    /observations/images/right_wrist → (T, H, W, C) uint8  (optional)
    /observations/qpos               → (T, state_dim) float
    /action                          → (T, action_dim) float
"""

import numpy as np
import torch
import h5py
import cv2

from data_utils.datasets.aloha_sim import AlohaSimDataset
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image


_CAM_KEY = {
    "top": "/observations/images/top",
    "left_wrist": "/observations/images/left_wrist",
    "right_wrist": "/observations/images/right_wrist",
}


class AlohaSimVideoDataset(AlohaSimDataset):
    """ALOHA Sim dataset that returns T-frame video sequences per sample.

    Inherits download logic, language instructions, and episode indexing
    from AlohaSimDataset.  Overrides ``__getitem__`` to batch-read
    *horizon* consecutive frames per camera with a single HDF5 fancy-index.
    """

    def __init__(
        self,
        dataset_path_list: list = [],
        camera_names: list = ["top"],
        chunk_size: int = 16,
        horizon: int = 16,
        frame_skip: int = 1,
        input_steps: int = 1,
        ctrl_space: str = "joint",
        ctrl_type: str = "abs",
        image_size: tuple = (256, 256),
        preload_data: bool = False,
    ):
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.input_steps = input_steps
        super().__init__(
            dataset_path_list, camera_names, chunk_size,
            ctrl_space, ctrl_type, image_size, preload_data,
        )

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        episode_len = self.episode_len[episode_id]

        root = (
            self.loaded_data[dataset_path]
            if self.loaded_data is not None
            else h5py.File(dataset_path, "r")
        )

        try:
            action_all = root["/action"][()]
            qpos_all = root["/observations/qpos"][()]

            end_ts = min(start_ts + self.chunk_size, episode_len)
            action = action_all[start_ts:end_ts].astype(np.float32)

            if self.ctrl_type == "delta":
                states_chunk = qpos_all[start_ts:end_ts].astype(np.float32)
                action = action - states_chunk
                state = states_chunk[0]
            else:
                state = qpos_all[start_ts].astype(np.float32)

            raw_indices = np.arange(self.horizon) * self.frame_skip + start_ts
            valid_mask = raw_indices < episode_len
            frame_indices = np.clip(raw_indices, 0, episode_len - 1)

            unique_indices, inverse = np.unique(frame_indices, return_inverse=True)

            all_cam_frames = []
            cam_list = ["top"]
            if "left_wrist" in self.camera_names:
                cam_list.append("left_wrist")
            if "right_wrist" in self.camera_names:
                cam_list.append("right_wrist")

            for cam in cam_list:
                unique_imgs = root[_CAM_KEY[cam]][unique_indices.tolist()]
                imgs = unique_imgs[inverse]  # (T, H, W, C)
                if self.image_size is not None:
                    h_t, w_t = self.image_size
                    if imgs.shape[1] != h_t or imgs.shape[2] != w_t:
                        imgs = np.stack(
                            [cv2.resize(fr, (w_t, h_t)) for fr in imgs]
                        )
                all_cam_frames.append(imgs)
        finally:
            if self.loaded_data is None and isinstance(root, h5py.File):
                root.close()

        padded_action = np.zeros(
            (self.chunk_size, action.shape[-1]), dtype=np.float32
        )
        actual_len = min(action.shape[0], self.chunk_size)
        padded_action[:actual_len] = action[:actual_len]
        is_pad_action = np.zeros(self.chunk_size, dtype=bool)
        is_pad_action[actual_len:] = True

        # (T, K, H, W, C) -> (T*K, C, H, W)
        video_np = np.stack(all_cam_frames, axis=1)
        video_np = ensure_uint8_image(video_np)
        video_tensor = torch.from_numpy(video_np)
        video_tensor = video_tensor.permute(0, 1, 4, 2, 3)  # (T, K, C, H, W)
        T, K = video_tensor.shape[:2]
        image_data = video_tensor.reshape(T * K, *video_tensor.shape[2:])

        is_pad_video = (~valid_mask).tolist()
        raw_lang = self.get_language_instruction()
        video_meta = build_video_meta(K, T, is_pad_video, self.input_steps, self.freq)

        return {
            "image": image_data,
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(padded_action).float(),
            "is_pad": torch.from_numpy(is_pad_action).bool(),
            "raw_lang": raw_lang,
            "reasoning": {"video": video_meta},
            "timestamp": start_ts,
            "episode_id": episode_id,
            "__index__": index,
        }
