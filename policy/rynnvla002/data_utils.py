"""
RynnVLA-002 data processing pipeline for ILStudio.

RynnVLA002Processor  – per-sample transform: ILStudio standard format →
                       intermediate format for on-GPU tokenization.
RynnVLA002Collator   – batch collation: group samples, keep images as lists.

The heavy tokenization (VQGAN for images, discretization for actions/states)
is intentionally deferred to the model's ``forward`` pass, which runs on GPU.
The processor and collator prepare lightweight, CPU-friendly data structures.
"""

import numpy as np
import torch
from loguru import logger


class RynnVLA002Processor:
    """Transform an ILStudio sample dict into the format expected by
    RynnVLA002Policy.forward().

    Input (ILStudio standard sample):
        image     – torch.Tensor (K, C, H, W) uint8
        state     – torch.Tensor (state_dim,) float  (already normalized)
        action    – torch.Tensor (chunk_size, action_dim) float  (already normalized)
        is_pad    – torch.Tensor (chunk_size,) bool
        raw_lang  – str

    Output (intermediate dict, ready for collation):
        images    – list[np.ndarray(H, W, 3) uint8]
                    ordered as [hist_front, hist_wrist, cur_front, cur_wrist]
        state     – np.ndarray (state_dim,) float32
        action    – np.ndarray (chunk_size, action_dim) float32
        is_pad    – np.ndarray (chunk_size,) bool
        raw_lang  – str
    """

    def __init__(
        self,
        num_views: int = 2,
        image_size: tuple = (256, 256),
        history_len: int = 2,
        with_wrist: bool = True,
        with_state: bool = True,
        time_horizon: int = 5,
    ):
        self.num_views = num_views
        self.image_size = image_size
        self.history_len = history_len
        self.with_wrist = with_wrist
        self.with_state = with_state
        self.time_horizon = time_horizon

    def __call__(self, sample):
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        K, C, H, W = image.shape
        images_hwc = []
        for k in range(K):
            img = np.transpose(image[k], (1, 2, 0))  # (C, H, W) -> (H, W, C)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            images_hwc.append(img)

        action = sample.get("action", None)
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            action = action.astype(np.float32)

        is_pad = sample.get("is_pad", None)
        if is_pad is not None:
            if isinstance(is_pad, torch.Tensor):
                is_pad = is_pad.numpy()
            if is_pad.ndim == 2:
                is_pad = is_pad[:, 0]

        state = sample.get("state", None)
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            state = state.astype(np.float32)

        raw_lang = sample.get("raw_lang", "")
        if isinstance(raw_lang, (list, np.ndarray)):
            raw_lang = str(raw_lang[0]) if len(raw_lang) > 0 else ""

        result = {
            "images": images_hwc,
            "raw_lang": str(raw_lang),
        }
        if action is not None:
            result["action"] = action
        if is_pad is not None:
            result["is_pad"] = is_pad
        if self.with_state and state is not None:
            result["state"] = state

        return result


class RynnVLA002Collator:
    """Collate processed RynnVLA-002 samples into a batch.

    Images are kept as lists of lists (not stacked into tensors) because
    they may have different token lengths after VQGAN encoding.
    Other fields are stacked into tensors.
    """

    def __call__(self, features):
        batch = {
            "images": [],
            "raw_lang": [],
        }
        has_action = "action" in features[0]
        has_state = "state" in features[0]
        has_is_pad = "is_pad" in features[0]

        actions, states, is_pads = [], [], []

        for f in features:
            batch["images"].append(f["images"])
            batch["raw_lang"].append(f["raw_lang"])
            if has_action:
                actions.append(f["action"])
            if has_state:
                states.append(f["state"])
            if has_is_pad:
                is_pads.append(f["is_pad"])

        if has_action:
            batch["action"] = torch.from_numpy(np.stack(actions))
        if has_state:
            batch["state"] = torch.from_numpy(np.stack(states))
        if has_is_pad:
            batch["is_pad"] = torch.from_numpy(np.stack(is_pads))

        return batch
