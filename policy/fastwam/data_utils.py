"""Data processing pipeline for FastWAM within ILStudio.

Converts ILStudio standard samples (image/state/action/is_pad/raw_lang)
into the dict format expected by FastWAM's ``build_inputs`` / ``infer_*``.
"""

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger

from data_utils.datasets.wm.video_meta import convert_image_to_video


class FastWAMDataProcessor:
    """Per-sample processor: ILStudio sample -> FastWAM-ready sample.

    Responsible for:
    - Image: reshape ``(T*K, C, H, W)`` video frames into ``(C, T, H, W)``
      or single-frame ``(K, C, H, W)`` into ``(C, 1, H, W)``, scale to [-1, 1].
    - State: keep as ``(state_dim,)`` tensor.
    - Action: keep as ``(chunk_size, action_dim)`` tensor.
    - Language: keep ``raw_lang`` for text-encoder encoding at inference.
    """

    def __init__(
        self,
        num_views: int = 1,
        image_size: int | tuple[int, int] | list[int] = (224, 224),
    ):
        self.num_views = num_views
        if isinstance(image_size, (list, tuple)):
            self._resize_size = (int(image_size[0]), int(image_size[1]))
        else:
            s = int(image_size)
            self._resize_size = (s, s)

    def _resize_chw(self, img: torch.Tensor) -> torch.Tensor:
        """Match training resolution (same as AlohaSimVideoDataset cv2.resize to image_size)."""
        if img.ndim != 3:
            return img
        th, tw = self._resize_size
        if th <= 0 or tw <= 0:
            return img
        _, h, w = img.shape
        if h == th and w == tw:
            return img
        x = img.unsqueeze(0)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def __call__(self, sample: dict) -> dict:
        out = {}

        image = sample.get("image")
        if image is not None:
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(np.array(image))
            if image.dtype == torch.uint8:
                image = image.float()
            if image.max() > 2.0:
                image = image * (2.0 / 255.0) - 1.0

            video_meta = None
            if "reasoning" in sample and isinstance(sample["reasoning"], dict):
                video_meta = sample["reasoning"].get("video")

            if video_meta is not None:
                frames = convert_image_to_video(image, video_meta, is_batch=False)
                T, K = int(frames.shape[0]), int(frames.shape[1])
                if K > 1:
                    resized = []
                    for k in range(K):
                        resized.append(
                            torch.stack([self._resize_chw(frames[t, k]) for t in range(T)])
                        )
                    frames = torch.cat(resized, dim=-1)
                else:
                    frames = torch.stack(
                        [self._resize_chw(frames[t, 0]) for t in range(T)]
                    )
                out["video"] = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
                out["_video_is_pad"] = video_meta.get("is_pad", None)
            elif image.ndim == 4:
                K = image.shape[0]
                if K > 1:
                    image = torch.cat([self._resize_chw(image[k]) for k in range(K)], dim=-1)
                else:
                    image = self._resize_chw(image[0])  # (C, H, W)
                out["image_single"] = image
            elif image.ndim == 3:
                out["image_single"] = self._resize_chw(image)
            else:
                out["image_single"] = image

        if "state" in sample and sample["state"] is not None:
            state = sample["state"]
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(np.array(state), dtype=torch.float32)
            out["state"] = state.float()

        if "action" in sample and sample["action"] is not None:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(np.array(action), dtype=torch.float32)
            out["action"] = action.float()

        if "is_pad" in sample and sample["is_pad"] is not None:
            is_pad = sample["is_pad"]
            if not isinstance(is_pad, torch.Tensor):
                is_pad = torch.as_tensor(np.array(is_pad))
            is_pad = is_pad.bool()
            # ILStudio is_pad may be (chunk_size, action_dim); reduce to (chunk_size,)
            if is_pad.ndim == 2:
                is_pad = is_pad.any(dim=-1)
            out["is_pad"] = is_pad

        if "raw_lang" in sample:
            out["raw_lang"] = sample["raw_lang"]
        if "reasoning" in sample:
            out["reasoning"] = sample["reasoning"]

        return out


class FastWAMDataCollator:
    """Batch collator: list of processed samples -> FastWAM batch dict.

    For training, produces the batch expected by ``FastWAM.build_inputs``:
        video: (B, C, T, H, W)
        action: (B, T_act, action_dim)
        context / context_mask: if cached text embeddings are available
        proprio: (B, 1, state_dim) if state is present

    For inference (via MetaPolicy), produces the batch for ``select_action``:
        image: (B, C, H, W) in [-1, 1]
        state: (B, state_dim)
        raw_lang: list of strings
    """

    def __init__(self, is_training: bool = True):
        self.is_training = is_training

    def __call__(self, instances: list[dict]) -> dict:
        if self.is_training:
            return self._collate_train(instances)
        return self._collate_inference(instances)

    def _collate_train(self, instances: list[dict]) -> dict:
        batch = {}

        if "video" in instances[0]:
            batch["video"] = torch.stack([s["video"] for s in instances])
        elif "image_single" in instances[0]:
            imgs = torch.stack([s["image_single"] for s in instances])
            batch["video"] = imgs.unsqueeze(2)

        if "action" in instances[0]:
            batch["action"] = torch.stack([s["action"] for s in instances])

        if "is_pad" in instances[0]:
            batch["action_is_pad"] = torch.stack([s["is_pad"] for s in instances])

        if "_video_is_pad" in instances[0] and instances[0]["_video_is_pad"] is not None:
            pads = []
            for s in instances:
                vp = s["_video_is_pad"]
                if not isinstance(vp, torch.Tensor):
                    vp = torch.as_tensor(vp)
                pads.append(vp.bool())
            batch["image_is_pad"] = torch.stack(pads)

        if "state" in instances[0]:
            states = torch.stack([s["state"] for s in instances])
            batch["proprio"] = states.unsqueeze(1)

        if "context" in instances[0]:
            batch["context"] = torch.stack([s["context"] for s in instances])
            batch["context_mask"] = torch.stack([s["context_mask"] for s in instances])
        elif "raw_lang" in instances[0]:
            batch["raw_lang"] = [s.get("raw_lang", "") for s in instances]

        return batch

    def _collate_inference(self, instances: list[dict]) -> dict:
        batch = {}

        if "image_single" in instances[0]:
            batch["image"] = torch.stack([s["image_single"] for s in instances])
        elif "video" in instances[0]:
            batch["image"] = torch.stack([s["video"][:, 0] for s in instances])

        if "state" in instances[0]:
            batch["state"] = torch.stack([s["state"] for s in instances])

        if "raw_lang" in instances[0]:
            batch["raw_lang"] = [s.get("raw_lang", "") for s in instances]

        if "context" in instances[0]:
            batch["context"] = torch.stack([s["context"] for s in instances])
            batch["context_mask"] = torch.stack([s["context_mask"] for s in instances])

        if "action" in instances[0]:
            batch["actions"] = torch.stack([s["action"] for s in instances])

        if "is_pad" in instances[0]:
            batch["is_pad"] = torch.stack([s["is_pad"] for s in instances])

        return batch
