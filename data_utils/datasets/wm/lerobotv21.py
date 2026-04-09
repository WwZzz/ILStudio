"""
LeRobot v2.1 Video Dataset – batch-decodes T video frames per camera in one pass.

Extends WrappedLerobotV21Dataset, overriding only ``__getitem__`` to decode all
horizon frames from each camera's video file in a single decode call instead of
T individual calls.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger

from benchmark.utils import resize_with_pad
from data_utils.datasets.lerobotv21_wrapper import (
    WrappedLerobotV21Dataset,
    decode_video_frames_torchcodec,
    decode_video_frames_torchvision,
)
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image


class LeRobotV21VideoDataset(WrappedLerobotV21Dataset):
    """LeRobot v2.1 dataset that returns T-frame video sequences per sample.

    Inherits metadata loading, parquet caching, statistics, and episode indexing
    from the parent class.  The key difference is that ``__getitem__`` batch-decodes
    *horizon* consecutive frames from each camera video in a single decode pass.
    """

    def __init__(
        self,
        dataset_path_list: List[str],
        camera_names: List[str] = [],
        root: Optional[str] = None,
        chunk_size: int = 16,
        ctrl_space: str = "ee",
        ctrl_type: str = "delta",
        image_size: Optional[Tuple[int, int]] = None,
        tolerance_s: float = 0.1,
        state_key: Union[str, List[str]] = "observation.state",
        action_key: Union[str, List[str]] = "action",
        episode_filter: Optional[dict] = None,
        download_videos: bool = True,
        filter_invalid_videos: bool = False,
        video_backend: Optional[str] = None,
        no_ensure_download: bool = False,
        local_only: bool = False,
        parquet_cache_size: int = 16,
        horizon: int = 16,
        frame_skip: int = 1,
        input_steps: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(
            dataset_path_list=dataset_path_list,
            camera_names=camera_names,
            root=root,
            chunk_size=chunk_size,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            image_size=image_size,
            tolerance_s=tolerance_s,
            state_key=state_key,
            action_key=action_key,
            episode_filter=episode_filter,
            download_videos=download_videos,
            filter_invalid_videos=filter_invalid_videos,
            video_backend=video_backend,
            no_ensure_download=no_ensure_download,
            local_only=local_only,
            parquet_cache_size=parquet_cache_size,
            *args,
            **kwargs,
        )
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.input_steps = input_steps

    def __getitem__(self, index: int) -> Dict[str, Any]:
        dataset_idx, ep_idx, frame_offset = self.index_to_sample_map[index]
        meta = self.dataset_metas[dataset_idx]

        parquet_data = self._load_episode_parquet(dataset_idx, ep_idx)

        # --- state ---
        state_data = self._get_data_by_keys(
            parquet_data, self.state_key, frame_idx=frame_offset
        )
        if state_data is not None:
            state = torch.tensor(state_data, dtype=torch.float32)
        else:
            state = torch.zeros(self.state_dim, dtype=torch.float32)

        # --- action chunk (pad with last action beyond episode end) ---
        action_full = self._get_data_by_keys(
            parquet_data, self.action_key, frame_idx=None
        )
        ep_len = len(action_full) if action_full is not None else 0

        if action_full is not None and ep_len > 0:
            end_idx = min(frame_offset + self.chunk_size, ep_len)
            valid_count = max(0, end_idx - frame_offset)

            if valid_count > 0:
                valid_actions = action_full[frame_offset:end_idx]
                if valid_count < self.chunk_size:
                    pad_count = self.chunk_size - valid_count
                    last_action = valid_actions[-1:] if len(valid_actions) > 0 else action_full[-1:]
                    padding = np.repeat(last_action, pad_count, axis=0)
                    actions = np.concatenate([valid_actions, padding], axis=0)
                    action_is_pad = np.array([False] * valid_count + [True] * pad_count)
                else:
                    actions = valid_actions
                    action_is_pad = np.array([False] * self.chunk_size)
            else:
                last_action = action_full[-1:]
                actions = np.repeat(last_action, self.chunk_size, axis=0)
                action_is_pad = np.array([True] * self.chunk_size)
        else:
            actions = np.zeros((self.chunk_size, self.action_dim))
            action_is_pad = np.array([True] * self.chunk_size)

        action = torch.tensor(actions, dtype=torch.float32)
        action_is_pad = torch.tensor(action_is_pad, dtype=torch.bool)

        # --- task / language ---
        task_idx = int(parquet_data.get("task_index", [0])[frame_offset])
        raw_lang = meta.tasks.get(task_idx, "")

        # --- timestamps for T horizon frames ---
        fps = meta.fps
        base_ts = float(parquet_data["timestamp"][frame_offset])

        valid_offsets = [frame_offset + t * self.frame_skip for t in range(self.horizon)]
        ts_ep_len = len(parquet_data.get("timestamp", []))
        is_pad_list = [off >= ts_ep_len for off in valid_offsets]

        last_valid_ts = float(parquet_data["timestamp"][min(ts_ep_len - 1, max(valid_offsets))])
        timestamps = []
        for t in range(self.horizon):
            if is_pad_list[t]:
                timestamps.append(last_valid_ts)
            else:
                timestamps.append(base_ts + t * self.frame_skip / fps)

        # --- batch-decode video frames for each camera ---
        cam_keys = meta.camera_keys if not self.camera_names else self.camera_names
        num_views = len(cam_keys)
        target_h, target_w = self._default_image_size

        all_cam_frames: List[torch.Tensor] = []
        for cam_key in cam_keys:
            frames = None

            if cam_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(ep_idx, cam_key)
                if video_path.exists():
                    try:
                        if self.video_backend == "torchcodec":
                            frames = decode_video_frames_torchcodec(
                                video_path, timestamps, self.tolerance_s, fps=fps
                            )
                        else:
                            frames = decode_video_frames_torchvision(
                                video_path, timestamps, self.tolerance_s
                            )
                    except Exception as e:
                        logger.warning(f"Failed to decode video {video_path}: {e}")

            if frames is None and cam_key in parquet_data:
                try:
                    single = self._load_image_from_parquet(
                        parquet_data, cam_key, frame_offset, meta
                    )
                    if single is not None:
                        frames = single.unsqueeze(0).expand(self.horizon, -1, -1, -1)
                except Exception as e:
                    logger.warning(f"Parquet fallback failed for {cam_key}: {e}")

            if frames is None:
                frames = torch.zeros(self.horizon, 3, target_h, target_w, dtype=torch.float32)
                if not hasattr(self, "_placeholder_warned"):
                    logger.warning(
                        f"Using placeholder (black) images for camera '{cam_key}' "
                        f"in episode {ep_idx}. Video file may be missing."
                    )
                    self._placeholder_warned = True

            if frames.shape[2] != target_h or frames.shape[3] != target_w:
                frames = resize_with_pad(frames, height=target_h, width=target_w)
                if isinstance(frames, np.ndarray):
                    frames = torch.from_numpy(frames).float()

            all_cam_frames.append(frames)  # each (T, C, H, W)

        # Stack cameras → (T, K, C, H, W) → reshape to (T*K, C, H, W)
        images = torch.stack(all_cam_frames, dim=1)  # (T, K, C, H, W)
        images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])

        images = (images * 255).clamp(0, 255).to(torch.uint8)
        images = ensure_uint8_image(images)

        episode_id = (
            int(self.per_dataset_episode_start[dataset_idx])
            + self.per_dataset_episodes[dataset_idx].index(ep_idx)
        )

        return {
            "image": images,
            "state": state,
            "action": action,
            "is_pad": action_is_pad,
            "raw_lang": raw_lang,
            "reasoning": {
                "video": build_video_meta(
                    num_views, self.horizon, is_pad_list, self.input_steps, self.freq
                )
            },
            "timestamp": frame_offset,
            "episode_id": episode_id,
            "__index__": index,
        }
