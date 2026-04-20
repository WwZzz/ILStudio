"""
RynnVLA-002 policy model for ILStudio.

Wrapper around the upstream RynnVLA-002 autoregressive VLA built on
Chameleon 7B with a continuous action head.  All core model code lives
in the vendored ``RynnVLA-002`` repository under
``policy/rynnvla002/RynnVLA-002/``.

Supports:
- Training with combined CE (discrete tokens) + L1 (continuous action head) loss
- Inference via ``select_action`` using the continuous action head
"""

import json
import os
import sys
import threading
import numpy as np
import torch
from pathlib import Path
from collections import deque
from loguru import logger
from transformers import PretrainedConfig, PreTrainedModel, GenerationConfig


# ======================================================================
# Submodule import setup
# ======================================================================
def _setup_rynnvla002_import_path() -> None:
    """Ensure the upstream RynnVLA-002 packages are importable.

    The vendor repo is at ``policy/rynnvla002/RynnVLA-002/``.
    We add both the repo root (for ``xllmx``) and the
    ``rynnvla-002/`` sub-directory (for ``model``, ``data``) to sys.path.
    """
    repo_root = Path(__file__).resolve().parent / "RynnVLA-002"
    sub_root = repo_root / "rynnvla-002"

    if not (sub_root / "model" / "__init__.py").is_file():
        raise ImportError(
            "RynnVLA-002 upstream is missing.  Clone it with:\n"
            "  git clone https://github.com/alibaba-damo-academy/RynnVLA-002.git "
            "policy/rynnvla002/RynnVLA-002\n"
            f"(expected {sub_root / 'model' / '__init__.py'})"
        ) from None

    for p in [str(repo_root), str(sub_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_rynnvla002_import_path()

from model import (
    ChameleonXLLMXConfig,
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)
from model import chameleon_vae_ori
from data.convertsation import Conversation
from xllmx.model.tokenizer import Tokenizer


# ======================================================================
# Config
# ======================================================================
class RynnVLA002PolicyConfig(PretrainedConfig):
    model_type = "rynnvla002"

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 8,
        time_horizon: int = 5,
        history_len: int = 2,
        with_state: bool = True,
        with_wrist: bool = True,
        image_size: list = None,
        max_seq_len: int = 4096,
        mask_image_logits: bool = True,
        dropout: float = 0.0,
        z_loss_weight: float = 0.0,
        loss_ct_weight: float = 1.0,
        model_size: str = "7B",
        pretrained_path: str = "",
        tokenizer_path: str = "",
        chameleon_tokenizer_dir: str = "",
        n_bins: int = 256,
        auto_download_ckpts: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.time_horizon = time_horizon
        self.history_len = history_len
        self.with_state = with_state
        self.with_wrist = with_wrist
        self.image_size = image_size or [256, 256]
        self.max_seq_len = max_seq_len
        self.mask_image_logits = mask_image_logits
        self.dropout = dropout
        self.z_loss_weight = z_loss_weight
        self.loss_ct_weight = loss_ct_weight
        self.model_size = model_size
        self.pretrained_path = pretrained_path
        self.tokenizer_path = tokenizer_path
        self.chameleon_tokenizer_dir = chameleon_tokenizer_dir
        self.n_bins = n_bins
        self.auto_download_ckpts = auto_download_ckpts


# ======================================================================
# Item Processor (on-GPU tokenization)
# ======================================================================
class _OnDeviceItemProcessor:
    """Tokenizes images (VQGAN), actions and states into token sequences
    on GPU.  Built as a thin wrapper so it can be instantiated lazily
    after the model is placed on device.
    """

    image_start_token = "<racm3:break>"
    image_end_token = "<eoss>"
    new_line_token = "<reserved08799>"
    # 必须与 ChameleonXLLMXForConditionalGeneration_ck_action_head 一致：
    # generate_att_mask_3 / find_sequences / decode_token_ids_to_actions / inference 末尾均使用 10004、15004。
    # 若误用 token2id("<reserved10000>")（多为 10000），训练序列里无 10004 → ActionHead 找不到块、
    # loss_ct 被置 0，且动作监督与 CE 语义与预训练不对齐。
    ACTION_BLOCK_START_ID = 10004
    ACTION_BLOCK_END_ID = 15004
    state_start_token = "<reserved15500>"
    state_end_token = "<reserved16000>"

    def __init__(self, tokenizer_path: str, chameleon_tokenizer_dir: str,
                 target_size: int = 256, device: str = "cuda"):
        self.device = device
        self.target_size = target_size

        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        media_symbols = ["<|image|>", "<|action|>", "<|state|>"]
        self.tokenizer.tokenizer.add_tokens(media_symbols)

        self.sep_token = "<reserved08706>"

        ckpt_dir = chameleon_tokenizer_dir
        vocab_json = os.path.join(ckpt_dir, "text_tokenizer.json")
        self.chameleon_vocab = chameleon_vae_ori.VocabInfo(
            json.load(open(vocab_json, encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_translation = chameleon_vae_ori.VocabTranslation(
            self.chameleon_vocab, device=device,
        )
        self.chameleon_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path=os.path.join(ckpt_dir, "vqgan.yaml"),
            ckpt_path=os.path.join(ckpt_dir, "vqgan.ckpt"),
            device=device,
        )

        self.patch_size = 32
        self.n_bins = 256
        self.bins = np.linspace(-1, 1, self.n_bins)

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    @torch.no_grad()
    def tokenize_image(self, pil_image):
        """Tokenize a PIL image through VQGAN → token ids."""
        from PIL import Image
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(np.array(pil_image).astype(np.uint8))

        w, h = pil_image.size
        if w != self.target_size or h != self.target_size:
            pil_image = pil_image.resize((self.target_size, self.target_size))

        w_grids = pil_image.size[0] // self.patch_size
        h_grids = pil_image.size[1] // self.patch_size

        image_toks = self.chameleon_translation.convert_img2bp2(
            self.chameleon_image_tokenizer.img_tokens_from_pil(pil_image)
        ).view(-1)

        full_image_toks = image_toks.reshape(
            pil_image.size[1] // 16, pil_image.size[0] // 16
        )
        new_line_id = self.token2id(self.new_line_token)
        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(
                    pil_image.size[1] // 16, 1,
                    device=full_image_toks.device, dtype=full_image_toks.dtype,
                ) * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]
        return result_toks

    def tokenize_action(self, action: np.ndarray):
        """Discretize a normalized [-1, 1] action into token ids."""
        action = np.clip(action, -1.0, 1.0)
        base = self.ACTION_BLOCK_START_ID
        discretized = np.digitize(action, self.bins) + base + 1
        return [base, *discretized.tolist(), self.ACTION_BLOCK_END_ID]

    def tokenize_state(self, state: np.ndarray):
        """Discretize a normalized [-1, 1] state into token ids."""
        state = np.clip(state, -1.0, 1.0)
        discretized = np.digitize(state, self.bins) + self.token2id(self.state_start_token) + 1
        return [
            self.token2id(self.state_start_token),
            *discretized.tolist(),
            self.token2id(self.state_end_token),
        ]

    def build_training_tokens(
        self,
        images,
        actions,
        state=None,
        raw_lang="",
        time_horizon=5,
        with_state=True,
    ):
        """Build input_ids and labels for one training sample.

        Parameters
        ----------
        images : list of PIL.Image
            Current + history images (front, wrist alternating).
        actions : np.ndarray  (time_horizon, action_dim) already in [-1, 1]
        state : np.ndarray  (state_dim,) already in [-1, 1], or None
        raw_lang : str
        time_horizon : int
        with_state : bool

        Returns
        -------
        input_ids : list[int]
        labels : list[int]
        """
        from PIL import Image

        num_images = len(images)
        if with_state and state is not None:
            human_val = (
                f"What action should the robot take to {raw_lang}?"
                + "<|state|>" * 1
                + "<|image|>" * num_images
            )
        else:
            human_val = (
                f"What action should the robot take to {raw_lang}?"
                + "<|image|>" * num_images
            )

        gpt_val = "<|action|>" * time_horizon

        conv = Conversation()
        conv.append_message(conv.roles[0], human_val)
        conv.append_message(conv.roles[1], gpt_val)
        result = conv.process()

        all_input_ids = []
        all_labels = []

        image_idx = 0
        action_idx = 0
        state_used = False

        bos_id = self.tokenizer.bos_id
        all_input_ids.append(bos_id)
        all_labels.append(-100)

        for piece in result["pieces"]:
            text = piece["data"]
            predict = piece["predict"]

            pos = 0
            while pos < len(text):
                if text[pos:].startswith("<|image|>"):
                    if image_idx < len(images):
                        img_toks = self.tokenize_image(images[image_idx])
                        image_idx += 1
                    else:
                        img_toks = []
                    all_input_ids.extend(img_toks)
                    if predict:
                        all_labels.extend(img_toks)
                    else:
                        all_labels.extend([-100] * len(img_toks))
                    pos += len("<|image|>")

                elif text[pos:].startswith("<|action|>"):
                    if action_idx < time_horizon:
                        act = actions[action_idx] if action_idx < len(actions) else actions[-1]
                        act_toks = self.tokenize_action(act)
                        action_idx += 1
                    else:
                        act_toks = []
                    all_input_ids.extend(act_toks)
                    if predict:
                        all_labels.extend(act_toks)
                    else:
                        all_labels.extend([-100] * len(act_toks))
                    pos += len("<|action|>")

                elif text[pos:].startswith("<|state|>"):
                    if not state_used and state is not None:
                        state_toks = self.tokenize_state(state)
                        state_used = True
                    else:
                        state_toks = []
                    all_input_ids.extend(state_toks)
                    all_labels.extend([-100] * len(state_toks))
                    pos += len("<|state|>")

                else:
                    end = len(text)
                    for marker in ["<|image|>", "<|action|>", "<|state|>"]:
                        idx = text.find(marker, pos)
                        if idx != -1:
                            end = min(end, idx)
                    segment = text[pos:end]
                    seg_toks = self.tokenizer.encode_segment(segment)
                    all_input_ids.extend(seg_toks)
                    if predict:
                        all_labels.extend(seg_toks)
                    else:
                        all_labels.extend([-100] * len(seg_toks))
                    pos = end

        return all_input_ids, all_labels

    def build_inference_tokens(self, images, state=None, raw_lang="",
                               with_state=True):
        """Build input_ids for inference (no labels, ends with action start token).

        Returns
        -------
        input_ids : list[int]
        """
        from PIL import Image

        num_images = len(images)
        if with_state and state is not None:
            human_val = (
                f"What action should the robot take to {raw_lang}?"
                + "<|state|>" * 1
                + "<|image|>" * num_images
            )
        else:
            human_val = (
                f"What action should the robot take to {raw_lang}?"
                + "<|image|>" * num_images
            )

        conv = Conversation()
        conv.append_message(conv.roles[0], human_val)
        conv.append_message(conv.roles[1], None)

        all_input_ids = [self.tokenizer.bos_id]

        prompt_text = conv.get_prompt()
        image_idx = 0
        state_used = False
        pos = 0
        while pos < len(prompt_text):
            if prompt_text[pos:].startswith("<|image|>"):
                if image_idx < len(images):
                    img_toks = self.tokenize_image(images[image_idx])
                    image_idx += 1
                else:
                    img_toks = []
                all_input_ids.extend(img_toks)
                pos += len("<|image|>")
            elif prompt_text[pos:].startswith("<|state|>"):
                if not state_used and state is not None:
                    state_toks = self.tokenize_state(state)
                    state_used = True
                else:
                    state_toks = []
                all_input_ids.extend(state_toks)
                pos += len("<|state|>")
            else:
                end = len(prompt_text)
                for marker in ["<|image|>", "<|state|>"]:
                    idx = prompt_text.find(marker, pos)
                    if idx != -1:
                        end = min(end, idx)
                segment = prompt_text[pos:end]
                seg_toks = self.tokenizer.encode_segment(segment)
                all_input_ids.extend(seg_toks)
                pos = end

        all_input_ids.append(10004)
        return all_input_ids


# ======================================================================
# Policy Model
# ======================================================================
class RynnVLA002Policy(PreTrainedModel):
    config_class = RynnVLA002PolicyConfig

    # ``nn.DataParallel`` 每步复制 replica，实例上的 ``_item_processor`` 会丢失；若每步重建则会
    # 反复从磁盘加载 VQGAN（上游 ``print('VQModel loaded...')``）。按 (device, 路径) 缓存一份。
    _item_processor_cache: dict[tuple, "_OnDeviceItemProcessor"] = {}
    _item_processor_lock = threading.Lock()
    _action_dim_mismatch_logged = False

    def __init__(self, config: RynnVLA002PolicyConfig):
        super().__init__(config)
        self.chameleon = None
        self._item_processor = None
        self._image_history = deque(maxlen=config.history_len)
        self._wrist_history = deque(maxlen=config.history_len)

    def _chameleon_device(self) -> torch.device:
        """Device for Chameleon / item processor.

        ``next(module.parameters())`` can raise ``StopIteration`` under
        ``nn.DataParallel`` when a replica exposes an empty parameter iterator;
        fall back to buffers or the active CUDA device.
        """
        if self.chameleon is None:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu")
        for p in self.chameleon.parameters(recurse=True):
            return p.device
        for b in self.chameleon.buffers(recurse=True):
            return b.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _ensure_chameleon(self):
        """Lazy-load the Chameleon backbone if not yet loaded."""
        if self.chameleon is not None:
            return
        cfg = self.config
        path = (getattr(cfg, "pretrained_path", None) or "").strip()
        if not path:
            raise ValueError(
                "`pretrained_path` is empty; cannot load the Chameleon backbone."
            )
        self.chameleon = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            path,
            action_dim=cfg.action_dim,
            time_horizon=cfg.time_horizon,
            max_position_embeddings=cfg.max_seq_len,
            mask_image_logits=cfg.mask_image_logits,
            dropout=cfg.dropout,
            z_loss_weight=cfg.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        if hasattr(self.chameleon.model, "vqmodel"):
            del self.chameleon.model.vqmodel
        logger.info("Loaded ChameleonXLLMX backbone")

    def _ensure_item_processor(self):
        """Lazy-init the on-device item processor (cached across DP replicas / steps)."""
        device = self._chameleon_device()
        key = (
            str(device),
            str(getattr(self.config, "tokenizer_path", "") or ""),
            str(getattr(self.config, "chameleon_tokenizer_dir", "") or ""),
            int(self.config.image_size[0]),
        )
        with RynnVLA002Policy._item_processor_lock:
            if key not in RynnVLA002Policy._item_processor_cache:
                logger.info("Initializing shared RynnVLA-002 item processor for {}", key[0])
                RynnVLA002Policy._item_processor_cache[key] = _OnDeviceItemProcessor(
                    tokenizer_path=self.config.tokenizer_path,
                    chameleon_tokenizer_dir=self.config.chameleon_tokenizer_dir,
                    target_size=self.config.image_size[0],
                    device=str(device),
                )
        self._item_processor = RynnVLA002Policy._item_processor_cache[key]

    def load_pretrained_chameleon(self, path: str):
        """Load pretrained Chameleon weights from a directory."""
        cfg = self.config
        self.chameleon = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            path,
            action_dim=cfg.action_dim,
            time_horizon=cfg.time_horizon,
            max_position_embeddings=cfg.max_seq_len,
            mask_image_logits=cfg.mask_image_logits,
            dropout=cfg.dropout,
            z_loss_weight=cfg.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        if hasattr(self.chameleon.model, "vqmodel"):
            del self.chameleon.model.vqmodel
        logger.info(f"Loaded ChameleonXLLMX backbone from {path}")

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch):
        """Training forward: tokenize on GPU, run Chameleon, return loss dict.

        Expected batch keys (from collator):
            images    – list[list[np.ndarray(H,W,3) uint8]]  (B, num_images)
            action    – torch.Tensor (B, chunk_size, action_dim) in [-1, 1]
            state     – torch.Tensor (B, state_dim) in [-1, 1]   (optional)
            raw_lang  – list[str]
            is_pad    – torch.Tensor (B, chunk_size) bool

        Under ``nn.DataParallel``, tensor fields are sharded on dim 0 but Python
        lists are replicated in full; we align list length to ``action.shape[0]``.
        """
        self._ensure_item_processor()
        proc = self._item_processor
        cfg = self.config
        device = self._chameleon_device()

        adim = int(batch["action"].shape[-1])
        if adim != int(cfg.action_dim) and not RynnVLA002Policy._action_dim_mismatch_logged:
            RynnVLA002Policy._action_dim_mismatch_logged = True
            logger.warning(
                "batch['action'] 最后一维为 {}，但 policy action_dim={}；"
                "会导致动作 token 与 ActionHead/find_sequences 不一致，CE 可能 nan。请对齐 task 与 configs/policy/rynnvla002.yaml。",
                adim,
                cfg.action_dim,
            )

        all_input_ids = []
        all_labels = []

        batch_size = int(batch["action"].shape[0])
        raw_langs = batch["raw_lang"]
        images_list = batch["images"]
        if isinstance(raw_langs, list) and len(raw_langs) > batch_size:
            raw_langs = raw_langs[:batch_size]
        if isinstance(images_list, list) and len(images_list) > batch_size:
            images_list = images_list[:batch_size]
        if len(raw_langs) != batch_size or len(images_list) != batch_size:
            raise ValueError(
                f"Batch size mismatch: action B={batch_size}, len(raw_lang)={len(raw_langs)}, "
                f"len(images)={len(images_list)}."
            )

        for i in range(batch_size):
            images_np = images_list[i]
            from PIL import Image
            pil_images = [Image.fromarray(img) for img in images_np]

            action = batch["action"][i].cpu().numpy()
            time_horizon = min(cfg.time_horizon, action.shape[0])
            action = action[:time_horizon]

            state = None
            if cfg.with_state and "state" in batch:
                state = batch["state"][i].cpu().numpy()

            raw_lang = raw_langs[i]

            input_ids, labels = proc.build_training_tokens(
                images=pil_images,
                actions=action,
                state=state,
                raw_lang=raw_lang,
                time_horizon=time_horizon,
                with_state=cfg.with_state,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        c_loss, additional_loss_dict, logits, hidden_states, labels_out, predicted_actions, loss_ct = (
            self.chameleon(
                input_ids=all_input_ids,
                labels=all_labels,
                output_hidden_states=True,
                training=True,
                att_mask=True,
            )
        )

        loss = c_loss + cfg.loss_ct_weight * loss_ct

        for key, (val, weight) in additional_loss_dict.items():
            loss = loss + val * weight

        return {
            "loss": loss,
            "ce_loss": c_loss.detach(),
            "ct_loss": loss_ct.detach() if isinstance(loss_ct, torch.Tensor) else loss_ct,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def reset(self):
        """Clear history buffers for a new episode."""
        self._image_history.clear()
        self._wrist_history.clear()

    @torch.no_grad()
    def select_action(self, batch_obs):
        """Run inference for ILStudio's MetaPolicy.

        Parameters
        ----------
        batch_obs : dict
            image     – (B, K, C, H, W) uint8 or float tensor, or (K, C, H, W)
            state     – (B, state_dim) or (state_dim,)
            raw_lang  – str or list[str]

        Returns
        -------
        actions : torch.Tensor  (B, time_horizon, action_dim) in [-1, 1]
        """
        self._ensure_item_processor()
        proc = self._item_processor
        cfg = self.config
        device = self._chameleon_device()
        self.chameleon.eval()

        image = batch_obs.get("image", batch_obs.get("images"))
        state = batch_obs.get("state", batch_obs.get("qpos"))
        raw_lang = batch_obs.get("raw_lang", "")
        if isinstance(raw_lang, list):
            raw_lang = raw_lang[0]

        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0).unsqueeze(0)
            elif image.dim() == 4:
                image = image.unsqueeze(0)

        if isinstance(state, torch.Tensor):
            if state.dim() == 1:
                state = state.unsqueeze(0)

        B = image.shape[0] if isinstance(image, torch.Tensor) else 1
        all_actions = []

        for b in range(B):
            if isinstance(image, torch.Tensor):
                imgs_tensor = image[b]
            else:
                imgs_tensor = image

            from PIL import Image as PILImage
            pil_imgs = []
            if isinstance(imgs_tensor, torch.Tensor):
                for k in range(imgs_tensor.shape[0]):
                    img_np = imgs_tensor[k].permute(1, 2, 0).cpu().numpy()
                    if img_np.max() <= 1.0 and img_np.dtype in (np.float32, np.float64):
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    pil_imgs.append(PILImage.fromarray(img_np))

            if cfg.with_wrist and len(pil_imgs) >= 2:
                cur_front = pil_imgs[0]
                cur_wrist = pil_imgs[1]
            elif len(pil_imgs) >= 1:
                cur_front = pil_imgs[0]
                cur_wrist = None
            else:
                raise ValueError("No images provided for inference")

            self._image_history.append(cur_front)
            if cur_wrist is not None:
                self._wrist_history.append(cur_wrist)

            inference_images = []
            if cfg.history_len >= 2 and len(self._image_history) >= 2:
                inference_images.append(list(self._image_history)[-2])
                if cfg.with_wrist and len(self._wrist_history) >= 2:
                    inference_images.append(list(self._wrist_history)[-2])
            inference_images.append(cur_front)
            if cfg.with_wrist and cur_wrist is not None:
                inference_images.append(cur_wrist)

            s = None
            if cfg.with_state and state is not None:
                s = state[b].cpu().numpy() if isinstance(state, torch.Tensor) else state

            tokens = proc.build_inference_tokens(
                images=inference_images,
                state=s,
                raw_lang=raw_lang,
                with_state=cfg.with_state,
            )

            input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)

            generation_config = GenerationConfig(
                max_new_tokens=1,
                max_length=cfg.max_seq_len,
                temperature=1.0,
                do_sample=False,
            )
            predicted = self.chameleon.generate_action_head(input_ids, generation_config)
            predicted = predicted.reshape(cfg.time_horizon, cfg.action_dim)
            all_actions.append(predicted.unsqueeze(0))

        return torch.cat(all_actions, dim=0)
