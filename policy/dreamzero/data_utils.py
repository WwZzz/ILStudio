"""
DreamZero data processing pipeline for ILStudio.

DreamZeroProcessor  – per-sample transform: ILStudio standard format →
                      DreamZero model input format
DreamZeroCollator   – batch collation with text tokenization (UMT5)
"""

import numpy as np
import os
import torch
import cv2
from loguru import logger


# ======================================================================
# Processor (sample-level)
# ======================================================================
class DreamZeroProcessor:
    """Transform an ILStudio sample dict into the format expected by
    DreamZero's WANPolicyHead.

    The WM dataset provides:
        image     – (T*K, C, H, W) uint8  (via reasoning["video"] metadata)
        state     – (state_dim,)
        action    – (chunk_size, action_dim)
        is_pad    – (chunk_size,) bool
        raw_lang  – str

    Output dict (per-sample, numpy, ready for collation):
        images          – (T, H_out, W_out, C) uint8
        text            – str
        text_negative   – str
        state           – (state_horizon, max_state_dim) float32
        state_mask      – (state_horizon, max_state_dim) bool
        action          – (action_horizon, max_action_dim) float32
        action_mask     – (action_horizon, max_action_dim) bool
        has_real_action – bool scalar
        embodiment_id   – int scalar
    """

    def __init__(
        self,
        max_action_dim: int = 64,
        max_state_dim: int = 64,
        action_horizon: int = 24,
        state_horizon: int = 16,
        num_video_frames: int = 33,
        image_size: tuple = (256, 256),
        num_views: int = 2,
        view_layout: str = "side_by_side",
        embodiment_id: int = 0,
        language_prefix: str = "",
    ):
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.action_horizon = action_horizon
        self.state_horizon = state_horizon
        self.num_video_frames = num_video_frames
        self.image_size = image_size
        self.num_views = num_views
        self.view_layout = view_layout
        self.embodiment_id = embodiment_id
        self.language_prefix = language_prefix

    def _concat_views(self, video_frames):
        """Concatenate camera views: (T, K, H, W, C) -> (T, H_out, W_out, C)"""
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.numpy()

        T, K, H, W, C = video_frames.shape
        if K == 1:
            return video_frames[:, 0]

        if self.view_layout == "side_by_side":
            return np.concatenate(
                [video_frames[:, i] for i in range(K)], axis=2
            )
        elif self.view_layout == "grid_2x2":
            out = np.zeros((T, 2 * H, 2 * W, C), dtype=video_frames.dtype)
            for i in range(min(K, 4)):
                r, c = divmod(i, 2)
                out[:, r * H : (r + 1) * H, c * W : (c + 1) * W, :] = video_frames[:, i]
            return out
        elif self.view_layout == "droid":
            out = np.zeros((T, 2 * H, 2 * W, C), dtype=video_frames.dtype)
            wrist = video_frames[:, min(2, K - 1)]
            wrist_wide = np.repeat(wrist, 2, axis=2)
            out[:, :H, :] = wrist_wide
            if K > 0:
                out[:, H:, :W] = video_frames[:, 0]
            if K > 1:
                out[:, H:, W:] = video_frames[:, 1]
            return out
        else:
            return np.concatenate(
                [video_frames[:, i] for i in range(K)], axis=2
            )

    def _resize_video(self, video):
        T, H, W, C = video.shape
        tgt_h, tgt_w = self.image_size
        if H == tgt_h and W == tgt_w:
            return video
        resized = np.stack([cv2.resize(video[t], (tgt_w, tgt_h)) for t in range(T)])
        return resized

    def _pad_to_dim(self, arr, target_dim, axis=-1):
        current = arr.shape[axis]
        if current >= target_dim:
            slices = [slice(None)] * len(arr.shape)
            slices[axis] = slice(0, target_dim)
            return arr[tuple(slices)]
        pad_width = [(0, 0)] * len(arr.shape)
        pad_width[axis] = (0, target_dim - current)
        return np.pad(arr, pad_width, constant_values=0)

    def _extract_video_from_wm_image(self, sample):
        """Reshape (T*K, C, H, W) image to (T, K, H, W, C) using reasoning["video"] metadata."""
        reasoning = sample.get("reasoning", {})
        if not isinstance(reasoning, dict):
            return None
        video_meta = reasoning.get("video")
        if video_meta is None:
            return None

        img = sample["image"]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if img.ndim != 4:
            return None

        T = video_meta["horizon"]
        K = video_meta["num_views"]
        video = img.reshape(T, K, *img.shape[1:])
        video = np.transpose(video, (0, 1, 3, 4, 2))  # (T, K, C, H, W) -> (T, K, H, W, C)
        return video

    def __call__(self, sample):
        # --- VIDEO ---
        if "video" in sample:
            video = sample["video"]
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            if video.ndim == 5:
                video = self._concat_views(video)
            video = self._resize_video(video)
        elif (wm_video := self._extract_video_from_wm_image(sample)) is not None:
            video = self._concat_views(wm_video)
            video = self._resize_video(video)
        else:
            img = sample["image"]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
            elif img.ndim == 4:
                imgs = [np.transpose(img[k], (1, 2, 0)) for k in range(img.shape[0])]
                img = np.concatenate(imgs, axis=1)
            video = np.stack([img] * self.num_video_frames, axis=0)

        T = video.shape[0]
        if T < self.num_video_frames:
            pad = np.stack([video[-1]] * (self.num_video_frames - T), axis=0)
            video = np.concatenate([video, pad], axis=0)
        elif T > self.num_video_frames:
            video = video[: self.num_video_frames]

        # --- STATE ---
        if "video_states" in sample:
            states = sample["video_states"]
            if isinstance(states, torch.Tensor):
                states = states.numpy()
        else:
            state = sample["state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            states = np.stack([state] * self.state_horizon, axis=0)

        if states.shape[0] < self.state_horizon:
            pad = np.stack(
                [states[-1]] * (self.state_horizon - states.shape[0]), axis=0
            )
            states = np.concatenate([states, pad], axis=0)
        elif states.shape[0] > self.state_horizon:
            states = states[: self.state_horizon]

        state_dim = states.shape[-1]
        states = self._pad_to_dim(states, self.max_state_dim, axis=-1).astype(np.float32)
        state_mask = np.zeros_like(states, dtype=bool)
        state_mask[:, :state_dim] = True

        # --- ACTION ---
        action = sample.get("action", None)
        is_pad = sample.get("is_pad", None)
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            act_len, act_dim = action.shape
            if act_len < self.action_horizon:
                pad_a = np.zeros(
                    (self.action_horizon - act_len, act_dim), dtype=np.float32
                )
                action = np.concatenate([action, pad_a], axis=0)
                if is_pad is not None:
                    if isinstance(is_pad, torch.Tensor):
                        is_pad = is_pad.numpy()
                    extra = np.ones(self.action_horizon - act_len, dtype=bool)
                    is_pad = np.concatenate([is_pad[:act_len], extra])
            elif act_len > self.action_horizon:
                action = action[: self.action_horizon]
                if is_pad is not None:
                    if isinstance(is_pad, torch.Tensor):
                        is_pad = is_pad.numpy()
                    is_pad = is_pad[: self.action_horizon]

            action = self._pad_to_dim(action, self.max_action_dim, axis=-1).astype(np.float32)
            action_mask = np.zeros_like(action, dtype=bool)
            action_mask[:, :act_dim] = True
            has_real_action = True
        else:
            action = np.zeros(
                (self.action_horizon, self.max_action_dim), dtype=np.float32
            )
            action_mask = np.zeros_like(action, dtype=bool)
            has_real_action = False

        # --- LANGUAGE ---
        raw_lang = sample.get("raw_lang", "")
        if isinstance(raw_lang, (list, np.ndarray)):
            raw_lang = str(raw_lang[0]) if len(raw_lang) > 0 else ""
        text = self.language_prefix + str(raw_lang).lower() if self.language_prefix else str(raw_lang)
        text_negative = (
            "Vibrant colors, overexposed, static, blurry details, text, "
            "subtitles, style, artwork, painting, image, still, grayscale, "
            "dull, worst quality, low quality, JPEG artifacts, ugly."
        )

        return {
            "images": video.astype(np.uint8),
            "text": text,
            "text_negative": text_negative,
            "state": states,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "has_real_action": np.array(has_real_action, dtype=bool),
            "embodiment_id": np.array(self.embodiment_id, dtype=np.int64),
        }


# ======================================================================
# Collator (batch-level)
# ======================================================================
class DreamZeroCollator:
    """Collate processed DreamZero samples into a batch.

    Handles text tokenization via UMT5 (or configurable) tokenizer
    and stacking of numpy arrays into tensors.
    """

    def __init__(
        self,
        tokenizer_path: str = "google/umt5-xxl",
        max_text_len: int = 512,
    ):
        self.tokenizer_path = tokenizer_path
        self.max_text_len = max_text_len
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            load_kwargs = {}
            if os.path.isdir(self.tokenizer_path):
                load_kwargs["local_files_only"] = True
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, **load_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load DreamZero tokenizer. "
                    f"tokenizer_path={self.tokenizer_path!r}. "
                    "Provide a local tokenizer directory via config.tokenizer_path "
                    "or the DREAMZERO_TOKENIZER_PATH environment variable."
                ) from exc
        return self._tokenizer

    def _tokenize(self, texts):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            add_special_tokens=True,
        )
        return encoded.input_ids, encoded.attention_mask

    def __call__(self, features):
        batch = {}
        keys = features[0].keys()

        for key in keys:
            if key == "text":
                texts = [f[key] for f in features]
                ids, mask = self._tokenize(texts)
                batch["text"] = ids
                batch["text_attention_mask"] = mask
            elif key == "text_negative":
                neg_texts = [f[key] for f in features]
                ids, mask = self._tokenize(neg_texts)
                batch["text_negative"] = ids
                batch["text_attention_mask_negative"] = mask
            elif key == "has_real_action":
                # Upstream WAN head expects shape (B,) for ``has_real_action[:, None] * loss``.
                # Collapse any per-timestep / malformed array to one bool per sample.
                flags = []
                for f in features:
                    v = f[key]
                    a = np.asarray(v, dtype=bool).reshape(-1)
                    flags.append(bool(a.any()) if a.size else False)
                batch[key] = torch.tensor(flags, dtype=torch.bool)
            else:
                values = [f[key] for f in features]
                if isinstance(values[0], np.ndarray):
                    batch[key] = torch.from_numpy(np.stack(values))
                elif isinstance(values[0], (int, float, bool, np.bool_)):
                    batch[key] = torch.tensor(values)
                elif isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values

        return batch
