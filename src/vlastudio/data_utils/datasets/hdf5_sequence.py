"""Temporal samples from VLAStudio's standard episode HDF5 format."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5SequenceDataset(Dataset):
    """Use image[K,T,C,H,W] for sequences, or image[K,C,H,W] for one frame.

    All offsets are relative to the sampled transition. Windows are clamped
    within one episode; padding masks distinguish repeated boundary frames.
    """
    def __init__(self, dataset_path_list, camera_names=("primary", "wrist"),
                 chunk_size=32, image_offsets=None, ctrl_space="ee", ctrl_type="delta",
                 image_size=None, preload_data=False):
        if len(dataset_path_list) != 1:
            raise ValueError("Pass one converted dataset root containing manifest.json")
        os.environ.setdefault("VLASTUDIO_FASTWAM_DATA", os.path.join(os.path.expanduser(os.environ.get("VLASTUDIO_CACHE", "~/.cache/vlastudio")), "data/fastwam/libero_hdf5"))
        self.dataset_dir = os.path.expandvars(os.path.expanduser(dataset_path_list[0]))
        self.root = Path(self.dataset_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("complete"):
            raise ValueError("Dataset conversion is incomplete")
        self.episodes = self.manifest["episodes"]
        self.episode_len = np.array([e["length"] for e in self.episodes])
        self.cumulative_len = np.cumsum(self.episode_len)
        self.num_episodes = len(self.episodes)
        self.camera_names = list(camera_names)
        self.chunk_size = int(chunk_size)
        self.image_offsets = list(image_offsets) if image_offsets is not None else [0]
        if not self.image_offsets or self.image_offsets[0] != 0 or self.chunk_size < 1:
            raise ValueError("Positive chunk_size and image offsets starting at zero are required")
        if any(a >= b for a, b in zip(self.image_offsets, self.image_offsets[1:])):
            raise ValueError("image_offsets must be strictly increasing")
        if image_size is not None or preload_data:
            raise ValueError("Resize in the policy processor; this dataset reads HDF5 on demand")
        self.ctrl_space, self.ctrl_type = ctrl_space, ctrl_type
        self.freq = self.manifest["fps"]

    def __len__(self):
        return int(self.cumulative_len[-1])

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_freq(self):
        return self.freq

    def extract_from_episode(self, episode_idx, keyname=()):
        paths = {"action": f"action_{self.ctrl_space}", "state": f"observations/state_{self.ctrl_space}"}
        with h5py.File(self.root / self.episodes[episode_idx]["path"], "r") as f:
            return {key: f[paths[key]][()] for key in keyname if key in paths}

    @staticmethod
    def _take(dataset, indices):
        unique, inverse = np.unique(indices, return_inverse=True)
        if dataset.ndim >= 3:
            # Read complete frame chunks to avoid the slow multidimensional
            # HDF5 selection path, then restore order and duplicate indices.
            frames = np.stack([dataset[int(index)] for index in unique])
            return frames[inverse]
        return dataset[unique][inverse]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.cumulative_len, index, side="right"))
        start = int(index - (self.cumulative_len[ep - 1] if ep else 0))
        length = int(self.episode_len[ep])
        action_indices = start + np.arange(self.chunk_size)
        image_indices = start + np.array(self.image_offsets)
        is_pad = action_indices >= length
        image_is_pad = image_indices >= length
        with h5py.File(self.root / self.episodes[ep]["path"], "r") as f:
            action = self._take(f[f"action_{self.ctrl_space}"], action_indices.clip(max=length-1)).copy()
            state = self._take(f[f"observations/state_{self.ctrl_space}"], action_indices.clip(max=length-1))
            images = np.stack([self._take(f[f"observations/image/{cam}"], image_indices.clip(max=length-1))
                               for cam in self.camera_names])
            language = f["language_instruction"].asstr()[0]
        if self.ctrl_type == "delta":
            action[is_pad, :-1] = 0  # Repeated gripper target holds the final opening.
        images = torch.from_numpy(images).permute(0, 1, 4, 2, 3)
        if len(self.image_offsets) == 1:
            images = images[:, 0]
        return dict(image=images, state=torch.from_numpy(state).float(),
                    action=torch.from_numpy(action).float(), is_pad=torch.from_numpy(is_pad),
                    image_is_pad=torch.from_numpy(image_is_pad),
                    raw_lang=language, reasoning="", timestamp=start, episode_id=ep, __index__=int(index))
