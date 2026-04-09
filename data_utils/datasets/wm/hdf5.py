"""
HDF5 Video Dataset – efficient batch frame reading for one-episode-per-file HDF5.

Expected HDF5 layout per file::

    /observations/images/<camera>  → (T, H, W, C) uint8
    /observations/qpos (or /state) → (T, state_dim) float
    /action (or /actions)          → (T, action_dim) float
"""

import numpy as np
import torch
import h5py
import cv2
from data_utils.datasets.base import EpisodicDataset
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image


class HDF5VideoDataset(EpisodicDataset):
    """Video dataset for standard ILStudio one-episode-per-file HDF5 layout.

    Extends EpisodicDataset with efficient batch HDF5 reads: a single
    fancy-index call loads all *horizon* frames per camera in one I/O op.
    """

    OBS_IMAGE_KEY = "/observations/images/{cam}"
    OBS_STATE_KEYS = ["/observations/qpos", "/state", "/observations/state"]
    ACTION_KEYS = ["/action", "/actions"]

    def __init__(
        self,
        dataset_path_list: list,
        camera_names: list = [],
        chunk_size: int = 16,
        horizon: int = 16,
        frame_skip: int = 1,
        input_steps: int = 1,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
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

    # ------------------------------------------------------------------
    # Key resolution helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, root, candidates):
        for k in candidates:
            if k in root:
                return k
        return None

    def _read_language(self, root):
        if "language_instruction" in root.attrs:
            lang = root.attrs["language_instruction"]
            return lang.decode("utf-8") if isinstance(lang, bytes) else str(lang)
        if "/language_instruction" in root:
            val = root["/language_instruction"][()]
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)
        return ""

    # ------------------------------------------------------------------
    # EpisodicDataset contracts
    # ------------------------------------------------------------------

    def get_language_instruction(self):
        return ""

    def load_feat_from_episode(self, dataset_path, feats=None):
        if feats is None:
            feats = []
        result = {}
        with h5py.File(dataset_path, "r") as root:
            for feat in feats:
                key = self._resolve_key(root, [f"/{feat}", f"/observations/{feat}"])
                if key is not None:
                    result[feat] = root[key][()].astype(np.float32)
        return result

    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        ep = self.loaded_data[dataset_path] if self.loaded_data is not None else None

        def _get(key):
            if ep is not None:
                return ep[key]
            with h5py.File(dataset_path, "r") as f:
                return f[key][()]

        def _find_key(candidates, src=None):
            if src is not None:
                for k in candidates:
                    if k in src:
                        return k
            else:
                with h5py.File(dataset_path, "r") as f:
                    for k in candidates:
                        if k in f:
                            return k
            return None

        state_key = _find_key(self.OBS_STATE_KEYS, ep)
        state = _get(state_key)[start_ts].astype(np.float32) if state_key else np.zeros(1, dtype=np.float32)

        action_key = _find_key(self.ACTION_KEYS, ep)
        action_all = _get(action_key)
        ep_len = action_all.shape[0]
        end_ts = min(start_ts + self.chunk_size, ep_len)
        action = action_all[start_ts:end_ts].astype(np.float32)

        image_dict = {}
        for cam in self.camera_names:
            img = _get(self.OBS_IMAGE_KEY.format(cam=cam))[start_ts]
            if self.image_size is not None:
                h, w = self.image_size
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))
            image_dict[cam] = img

        if ep is not None:
            lang = ""
            if "/language_instruction" in ep:
                val = ep["/language_instruction"]
                if isinstance(val, np.ndarray):
                    val = val.item()
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                lang = str(val)
        else:
            with h5py.File(dataset_path, "r") as f:
                lang = self._read_language(f)

        return {
            "action": action,
            "image": image_dict,
            "state": state,
            "language_instruction": lang,
        }

    # ------------------------------------------------------------------
    # Core __getitem__ – batch HDF5 read for T frames
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        episode_len = self.episode_len[episode_id]

        data_dict = self.load_onestep_from_episode(dataset_path, start_ts)
        action = data_dict["action"]
        state = data_dict["state"]
        raw_lang = data_dict["language_instruction"]
        reasoning = data_dict.get("reasoning", "")

        padded_action = np.zeros((self.chunk_size, action.shape[-1]), dtype=np.float32)
        actual_len = min(action.shape[0], self.chunk_size)
        padded_action[:actual_len] = action[:actual_len]
        is_pad_action = np.zeros(self.chunk_size, dtype=bool)
        is_pad_action[actual_len:] = True

        raw_indices = np.arange(self.horizon) * self.frame_skip + start_ts
        valid_mask = raw_indices < episode_len
        frame_indices = np.clip(raw_indices, 0, episode_len - 1)

        ep = self.loaded_data[dataset_path] if self.loaded_data is not None else None

        all_cam_frames = []
        for cam in self.camera_names:
            img_key = self.OBS_IMAGE_KEY.format(cam=cam)
            if ep is not None:
                all_imgs = ep[img_key]
            else:
                with h5py.File(dataset_path, "r") as f:
                    all_imgs = f[img_key][()]
            frames = all_imgs[frame_indices]
            if self.image_size is not None:
                h_t, w_t = self.image_size
                if frames.shape[1] != h_t or frames.shape[2] != w_t:
                    frames = np.stack([cv2.resize(fr, (w_t, h_t)) for fr in frames])
            all_cam_frames.append(frames)

        if all_cam_frames:
            video_np = np.stack(all_cam_frames, axis=1)  # (T, K, H, W, C)
            video_np = ensure_uint8_image(video_np)
            video_tensor = torch.from_numpy(video_np)
            video_tensor = video_tensor.permute(0, 1, 4, 2, 3)  # (T, K, C, H, W)
            T, K = video_tensor.shape[:2]
            image_data = video_tensor.reshape(T * K, *video_tensor.shape[2:])
        else:
            image_data = None
            K = 0
            T = self.horizon

        is_pad_video = (~valid_mask).tolist()
        video_meta = build_video_meta(K, T, is_pad_video, self.input_steps, self.freq)
        reasoning_dict = reasoning if isinstance(reasoning, dict) else {}
        reasoning_dict["video"] = video_meta

        return {
            "image": image_data,
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(padded_action).float(),
            "is_pad": torch.from_numpy(is_pad_action).bool(),
            "raw_lang": raw_lang,
            "reasoning": reasoning_dict,
            "timestamp": start_ts,
            "episode_id": episode_id,
            "__index__": index,
        }
