from dataclasses import dataclass, field
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List, Union, Optional, Tuple
from .paligemma_with_expert import PaliGemmaForConditionalGeneration, PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float64 if device == "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def resize_with_pad(img, width, height, pad_value=-1):
    """Resize image with padding to maintain aspect ratio."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Pad vector to new dimension."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


@dataclass
class NormalizationMode:
    """Normalization mode constants."""
    IDENTITY = "identity"
    MEAN_STD = "mean_std"


@dataclass
class FeatureType:
    """Feature type constants."""
    VISUAL = "visual"
    STATE = "state"
    ACTION = "action"


@dataclass
class PolicyFeature:
    """Policy feature definition."""
    type: str
    shape: tuple


class PI0FlowMatchingConfig(PretrainedConfig):
    model_type = "pi0-flow-matching"
    
    def __init__(
        self,
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        
        # Normalization mapping
        normalization_mapping: Dict[str, str] = None,
        
        # Padding dimensions
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        
        # Image preprocessing
        resize_imgs_with_padding: tuple = (224, 224),
        
        # Empty cameras
        empty_cameras: int = 0,
        
        # Tokenizer
        tokenizer_max_length: int = 48,
        
        # Decoding
        num_steps: int = 10,
        
        # Attention utils
        use_cache: bool = True,
        
        # Finetuning settings
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = False,
        train_state_proj: bool = True,
        attention_implementation: str = "fa2",
        
        # Training presets
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        
        scheduler_warmup_steps: int = 1000,
        scheduler_decay_steps: int = 30000,
        scheduler_decay_lr: float = 2.5e-6,
        
        # Task parameters
        state_dim: int = 14,
        action_dim: int = 14,
        camera_names: List[str] = None,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Input / output structure
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        
        # Normalization mapping
        if normalization_mapping is None:
            normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }
        self.normalization_mapping = normalization_mapping
        
        # Padding dimensions
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.proj_width = proj_width
        
        # Image preprocessing
        self.resize_imgs_with_padding = resize_imgs_with_padding
        
        # Empty cameras
        self.empty_cameras = empty_cameras
        
        # Tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        
        # Decoding
        self.num_steps = num_steps
        
        # Attention utils
        self.use_cache = use_cache
        
        # Finetuning settings
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.train_state_proj = train_state_proj
        self.attention_implementation = attention_implementation
        
        # Training presets
        self.optimizer_lr = optimizer_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_grad_clip_norm = optimizer_grad_clip_norm
        
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_steps = scheduler_decay_steps
        self.scheduler_decay_lr = scheduler_decay_lr
        
        # Task parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.camera_names = camera_names if camera_names is not None else ['primary']
        
        # Initialize input/output features
        self.input_features = {}
        self.output_features = {}
        
        # Add image features
        for i, camera_name in enumerate(self.camera_names):
            self.input_features[f"observation.images.{camera_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640)
            )
        
        # Add empty camera features if needed
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640)
            )
        
        # Add state feature
        self.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.state_dim,)
        )
        
        # Add action feature
        self.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(self.action_dim,)
        )
        
        # Add language feature
        self.input_features["task"] = PolicyFeature(
            type="text",
            shape=()
        )
    
    def validate_features(self) -> None:
        """Validate feature configuration."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 480, 640)
                )
    
    @property
    def image_features(self) -> List[str]:
        """Get list of image feature names."""
        return [k for k, v in self.input_features.items() if v.type == FeatureType.VISUAL]
    
    @property
    def action_feature(self) -> PolicyFeature:
        """Get action feature."""
        return self.output_features["action"]


class PI0FlowMatching(PreTrainedModel):
    config_class = PI0FlowMatchingConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        config.validate_features()

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            attention_implementation=config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config)

        # 构建投影层
        self.state_proj = nn.Linear(config.max_state_dim, config.proj_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(config.proj_width * 2, config.proj_width)
        self.action_time_mlp_out = nn.Linear(config.proj_width, config.proj_width)

        # Initialize tokenizer
        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        
        # Initialize queues for action prediction
        self._queues = {}
        self.reset()

        self.post_init()

    def reset(self):
        """Reset the policy state."""
        self._queues = {
            "action": [],
        }

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks, att_masks = [], [], []
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            img_mask = img_mask[:, None].expand(img_emb.shape[0], img_emb.shape[1])
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * img_emb.shape[1]
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(embs.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        embs, pad_masks, att_masks = [], [], []
        state_emb = self.state_proj(state).to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        pad_masks.append(torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state_emb.device))
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=state_emb.device
        ).type(dtype=state_emb.dtype)
        action_emb = self.action_in_proj(noisy_actions)
        action_time_emb = torch.cat([action_emb, time_emb[:, None, :]], dim=2)
        action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
        embs.append(action_time_emb)
        pad_masks.append(torch.ones(action_time_emb.shape[0], action_time_emb.shape[1], dtype=torch.bool, device=state_emb.device))
        att_masks += [1] + [0] * (self.config.n_action_steps - 1)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        return embs, pad_masks, att_masks

    def forward(self, batch: Dict[str, torch.Tensor], noise=None, time=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass for training."""
        # Prepare batch
        batch = self._prepare_batch(batch)
        
        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        
        # Forward through model
        loss_dict = self._forward_flow_matching(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        
        # Extract loss
        loss = loss_dict["loss"]
        
        return loss, loss_dict

    def _forward_flow_matching(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, is_pad=None):
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        
        loss_dict = {}
        loss_dict["losses_after_forward"] = losses.clone()
        if is_pad is not None:
            in_episode_bound = ~is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()
        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()
        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["loss"] = loss.item()
        return loss_dict

    def predict_action_chunk(self, batch: Dict[str, torch.Tensor], noise: torch.Tensor = None) -> torch.Tensor:
        """Predict action chunk for inference."""
        self.eval()
        batch = self._prepare_batch(batch)
        
        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        
        # Sample actions
        actions = self.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise)
        
        # Unpad actions
        original_action_dim = self.config.action_dim
        actions = actions[:, :, :original_action_dim]
        
        return actions

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for processing."""
        # Add any preprocessing here
        return batch

    def prepare_images(self, batch):
        """Prepare images for processing."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
            
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
            
        return images, img_masks

    def prepare_language(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the text input."""
        device = batch["observation.state"].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch["observation.state"].shape[0])]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state to max_state_dim."""
        state = batch["observation.state"][:, -1, :] if batch["observation.state"].ndim > 2 else batch["observation.state"]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action to max_action_dim."""
        actions = pad_vector(batch["action"], self.config.max_action_dim)
        return actions

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
    

def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype
    

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)