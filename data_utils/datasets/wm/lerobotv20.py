import numpy as np
import torch
from loguru import logger

from data_utils.datasets.lerobotv20_wrapper import (
    WrappedLerobotV20Dataset,
    decode_video_frames_torchcodec,
    decode_video_frames_torchvision,
)
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image
from benchmark.utils import resize_with_pad


class LeRobotV20VideoDataset(WrappedLerobotV20Dataset):

    def __init__(self, *args, horizon=16, frame_skip=1, input_steps=1, **kwargs):
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.input_steps = input_steps
        super().__init__(*args, **kwargs)

    def _decode_video_batch(self, video_path, timestamps, fps):
        if self.video_backend == "torchcodec":
            try:
                return decode_video_frames_torchcodec(
                    video_path, timestamps, self.tolerance_s, fps=fps
                )
            except Exception as e:
                logger.warning(
                    f"torchcodec failed for {video_path}: {e}. Falling back to pyav."
                )
                return decode_video_frames_torchvision(
                    video_path, timestamps, self.tolerance_s, backend="pyav"
                )
        return decode_video_frames_torchvision(
            video_path, timestamps, self.tolerance_s, backend="pyav"
        )

    def __getitem__(self, index):
        dataset_idx, ep_idx, frame_offset = self.index_to_sample_map[index]
        meta = self.dataset_metas[dataset_idx]
        parquet_data = self._load_episode_parquet(dataset_idx, ep_idx)

        ep_timestamps = parquet_data.get("timestamp")
        anchor_ts = float(ep_timestamps[frame_offset])
        ep_len = len(ep_timestamps)

        state_data = self._get_data_by_keys(
            parquet_data, self.state_key, frame_idx=frame_offset
        )
        state = (
            torch.tensor(state_data, dtype=torch.float32)
            if state_data is not None
            else torch.zeros(self.state_dim, dtype=torch.float32)
        )

        action_full = self._get_data_by_keys(
            parquet_data, self.action_key, frame_idx=None
        )
        if action_full is not None and len(action_full) > 0:
            end_idx = min(frame_offset + self.chunk_size, len(action_full))
            valid_count = max(0, end_idx - frame_offset)
            if valid_count > 0:
                valid_actions = action_full[frame_offset:end_idx]
                if valid_count < self.chunk_size:
                    pad_count = self.chunk_size - valid_count
                    padding = np.repeat(valid_actions[-1:], pad_count, axis=0)
                    actions = np.concatenate([valid_actions, padding], axis=0)
                    action_is_pad = np.array(
                        [False] * valid_count + [True] * pad_count
                    )
                else:
                    actions = valid_actions
                    action_is_pad = np.array([False] * self.chunk_size)
            else:
                actions = np.repeat(action_full[-1:], self.chunk_size, axis=0)
                action_is_pad = np.array([True] * self.chunk_size)
        else:
            actions = np.zeros((self.chunk_size, self.action_dim))
            action_is_pad = np.array([True] * self.chunk_size)

        action = torch.tensor(actions, dtype=torch.float32)
        action_is_pad = torch.tensor(action_is_pad, dtype=torch.bool)

        task_idx = int(parquet_data.get("task_index", [0])[frame_offset])
        raw_lang = meta.tasks.get(task_idx, "")

        # T timestamps for the video horizon
        frame_offsets = np.arange(self.horizon) * self.frame_skip + frame_offset
        video_is_pad = (frame_offsets >= ep_len).tolist()
        clamped_offsets = np.clip(frame_offsets, 0, ep_len - 1)
        timestamps = [float(ep_timestamps[o]) for o in clamped_offsets]

        cam_keys = self._get_requested_camera_keys(meta)
        target_h, target_w = self._default_image_size
        cam_frames = []

        for cam_key in cam_keys:
            frames = None

            if cam_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(ep_idx, cam_key)
                if video_path.exists():
                    try:
                        frames = self._decode_video_batch(
                            video_path, timestamps, meta.fps
                        )
                    except Exception as e:
                        logger.warning(f"Failed to decode video for {cam_key}: {e}")

            if frames is None and cam_key not in meta.video_keys:
                if (
                    cam_key not in parquet_data
                    and cam_key in self._parquet_available_columns[dataset_idx]
                ):
                    parquet_data = {
                        **parquet_data,
                        **self._load_episode_parquet(
                            dataset_idx, ep_idx,
                            columns=[cam_key], cache_result=False,
                        ),
                    }
                if cam_key in parquet_data:
                    frame_list = []
                    for off in clamped_offsets:
                        f = self._load_image_from_parquet(
                            parquet_data, cam_key, int(off), meta
                        )
                        if f is None:
                            f = torch.zeros(3, target_h, target_w)
                        frame_list.append(f)
                    frames = torch.stack(frame_list)

            if frames is None:
                frames = torch.zeros(self.horizon, 3, target_h, target_w)

            if frames.shape[2] != target_h or frames.shape[3] != target_w:
                frames = torch.cat(
                    [
                        resize_with_pad(f.unsqueeze(0), height=target_h, width=target_w)
                        for f in frames
                    ],
                    dim=0,
                )

            cam_frames.append(frames)

        # (K, T, C, H, W) -> (T, K, C, H, W) -> (T*K, C, H, W)
        images = torch.stack(cam_frames, dim=0)
        images = images.permute(1, 0, 2, 3, 4)
        num_views = images.shape[1]
        images = images.reshape(-1, *images.shape[2:])
        images = ensure_uint8_image(images)

        freq = meta.fps / self.frame_skip
        video_meta = build_video_meta(
            num_views, self.horizon, video_is_pad, self.input_steps, freq
        )

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
            "reasoning": {"video": video_meta},
            "timestamp": frame_offset,
            "episode_id": episode_id,
            "__index__": index,
        }
