"""
Video dataset implementations for temporal video sequence loading.

Each submodule implements efficient batch video loading for a specific data
format, reusing the same data sources as the corresponding image-only datasets
but reading consecutive frames at the I/O level.

Sample image format:  (T * num_views, C, H, W)  uint8
Video metadata:       reasoning["video"] = { num_views, horizon, is_pad, freq }

Use :func:`convert_image_to_video` / :func:`flat_image_to_video_btchw` to recover
``(T, K, C, H, W)`` or ``(B, T, C, H, W)`` without reimplementing layout math.
"""

from .video_meta import (
    build_video_meta,
    convert_image_to_video,
    flat_image_to_video_btchw,
)
from .hdf5 import HDF5VideoDataset
from .libero import LiberoVideoDataset
from .aloha_sim import AlohaSimVideoDataset
from .lerobotv21 import LeRobotV21VideoDataset
from .lerobotv20 import LeRobotV20VideoDataset
from .lerobot import LeRobotVideoDataset

__all__ = [
    "HDF5VideoDataset",
    "LiberoVideoDataset",
    "AlohaSimVideoDataset",
    "LeRobotV21VideoDataset",
    "LeRobotV20VideoDataset",
    "LeRobotVideoDataset",
    "build_video_meta",
    "flat_image_to_video_btchw",
    "convert_image_to_video",
]
