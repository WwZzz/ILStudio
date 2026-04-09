import torch
from loguru import logger

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset
from .video_meta import build_video_meta
from data_utils.utils import ensure_uint8_image
from benchmark.utils import resize_with_pad


class LeRobotVideoDataset(WrappedLerobotDataset):

    def __init__(self, *args, horizon=16, frame_skip=1, input_steps=1, **kwargs):
        self.horizon = horizon
        self.frame_skip = frame_skip
        self.input_steps = input_steps
        self._init_tolerance_s = kwargs.get("tolerance_s", 1e-3)
        super().__init__(*args, **kwargs)

        for i, (dataset, ds_meta) in enumerate(
            zip(self.datasets, self.dataset_metas)
        ):
            cam_timestamps = [
                t * self.frame_skip / ds_meta.fps for t in range(self.horizon)
            ]
            new_delta = {
                self._primary_action_key: [
                    t / ds_meta.fps for t in range(self.chunk_size)
                ]
            }
            for cam_key in ds_meta.camera_keys:
                new_delta[cam_key] = cam_timestamps

            self.datasets[i] = LeRobotDataset(
                self.dataset_path_list[i],
                root=self.root,
                delta_timestamps=new_delta,
                tolerance_s=self._init_tolerance_s,
                episodes=dataset.episodes if hasattr(dataset, "episodes") else None,
                video_backend=self.video_backend,
            )
            self.datasets[i] = self._optimize_dataset_columns(
                self.datasets[i], ds_meta, self.camera_names
            )

    def __getitem__(self, index):
        dataset_idx, start_ts = self._locate_dataset_for_transition(index)
        sample = self.datasets[dataset_idx][start_ts]

        action = self._get_data_from_sample(sample, self.action_key)
        state = self._get_data_from_sample(sample, self.state_key)
        raw_lang = sample["task"]
        episode_id = (
            self.per_dataset_episode_start[dataset_idx]
            + sample["episode_index"].item()
        )

        if "frame_index" in sample:
            timestamp = sample["frame_index"].item()
        elif "index" in sample:
            timestamp = sample["index"].item()
        else:
            timestamp = start_ts

        pad_key = f"{self._primary_action_key}_is_pad"
        if pad_key in sample:
            is_pad = sample[pad_key]
        elif "action_is_pad" in sample:
            is_pad = sample["action_is_pad"]
        else:
            is_pad = torch.zeros(action.shape[0], dtype=torch.bool)

        ds_meta = self.datasets[dataset_idx].meta
        all_camera_keys = ds_meta.camera_keys
        cam_keys = (
            [k for k in self.camera_names if k in all_camera_keys]
            if len(self.camera_names) > 0
            else list(all_camera_keys)
        )

        cam_frames = []
        video_is_pad = None

        for cam_key in cam_keys:
            if cam_key not in sample:
                logger.warning(f"Camera key '{cam_key}' not in sample, skipping")
                continue

            frames = sample[cam_key]  # (T, C, H, W) float from lerobot
            if self.image_size is not None:
                frames = torch.cat(
                    [
                        resize_with_pad(
                            f.unsqueeze(0),
                            height=self.image_size[1],
                            width=self.image_size[0],
                        )
                        for f in frames
                    ],
                    dim=0,
                )
            cam_frames.append(frames)

            if video_is_pad is None:
                cam_pad_key = f"{cam_key}_is_pad"
                if cam_pad_key in sample:
                    video_is_pad = sample[cam_pad_key].tolist()

        if video_is_pad is None:
            video_is_pad = [False] * self.horizon

        if not cam_frames:
            logger.warning(f"No camera images found for index {index}")
            cam_frames = [torch.zeros(self.horizon, 3, 224, 224)]

        # list of (T, C, H, W) -> (K, T, C, H, W) -> (T, K, C, H, W) -> (T*K, C, H, W)
        images = torch.stack(cam_frames, dim=0)
        images = images.permute(1, 0, 2, 3, 4)
        num_views = images.shape[1]
        images = images.reshape(-1, *images.shape[2:])
        images = ensure_uint8_image(images)

        freq = self.freq / self.frame_skip
        video_meta = build_video_meta(
            num_views, self.horizon, video_is_pad, self.input_steps, freq
        )

        return {
            "image": images,
            "state": state,
            "action": action,
            "is_pad": is_pad,
            "raw_lang": raw_lang,
            "reasoning": {"video": video_meta},
            "timestamp": timestamp,
            "episode_id": episode_id,
            "__index__": index,
        }
