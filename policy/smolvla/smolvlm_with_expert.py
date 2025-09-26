# SmolVLM with Expert adapted from LeRobot to IL-Studio
import copy
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    """Calculate intermediate size for FFN."""
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    """SmolVLM model with action expert - adapted from LeRobot."""
    
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
    ):
        super().__init__()
        
        if load_vlm_weights:
            print(f"Loading {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
            
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
            
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        
        # Create smaller LM expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of VLM layers ({len(self.get_vlm_model().text_model.layers)}) "
                f"must be divisible by num_expert_layers ({num_expert_layers})"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
            
        self.lm_expert = AutoModel.from_config(lm_expert_config)
        
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        """Get the inner VLM model."""
        return self.vlm.model

    def set_requires_grad(self):
        """Set gradient requirements for different components."""
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        """Embed images using the vision encoder."""
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed language tokens."""
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.FloatTensor]]]:
        """Forward pass through the model - simplified version for IL-Studio."""
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
            
        prefix_embeds, suffix_embeds = inputs_embeds
        
        if prefix_embeds is not None and suffix_embeds is not None:
            # Full forward pass
            return self._forward_full(prefix_embeds, suffix_embeds, attention_mask, position_ids, use_cache)
        elif prefix_embeds is not None:
            # Prefix only (for caching)
            return self._forward_prefix(prefix_embeds, attention_mask, position_ids, use_cache, fill_kv_cache)
        elif suffix_embeds is not None:
            # Suffix only (for generation)
            return self._forward_suffix(suffix_embeds, attention_mask, position_ids, past_key_values, use_cache)
        else:
            raise ValueError("At least one of prefix_embeds or suffix_embeds must be provided")

    def _forward_full(self, prefix_embeds, suffix_embeds, attention_mask, position_ids, use_cache):
        """Full forward pass through both VLM and expert."""
        # Process prefix through VLM
        vlm_outputs = self.get_vlm_model().text_model(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask[:, :prefix_embeds.shape[1]],
            position_ids=position_ids[:, :prefix_embeds.shape[1]],
            use_cache=False,
        )
        
        # Process suffix through expert text model  
        expert_outputs = self.lm_expert(
            inputs_embeds=suffix_embeds,
            attention_mask=attention_mask[:, prefix_embeds.shape[1]:],
            position_ids=position_ids[:, prefix_embeds.shape[1]:],
            use_cache=use_cache,
        )
        
        # Combine outputs - handle different output types
        def _get_hidden_states(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                return outputs.hidden_states
            elif isinstance(outputs, (tuple, list)):
                return outputs[0]
            else:
                return outputs
        
        vlm_hidden = _get_hidden_states(vlm_outputs)
        expert_hidden = _get_hidden_states(expert_outputs)
        combined_hidden_states = torch.cat([vlm_hidden, expert_hidden], dim=1)
        
        return [combined_hidden_states], getattr(expert_outputs, 'past_key_values', None)

    def _forward_prefix(self, prefix_embeds, attention_mask, position_ids, use_cache, fill_kv_cache):
        """Forward pass for prefix (caching)."""
        vlm_outputs = self.get_vlm_model().text_model(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        
        hidden_states = getattr(vlm_outputs, 'last_hidden_state', vlm_outputs)
        past_kv = getattr(vlm_outputs, 'past_key_values', None) if fill_kv_cache else None
        
        return [hidden_states], past_kv

    def _forward_suffix(self, suffix_embeds, attention_mask, position_ids, past_key_values, use_cache):
        """Forward pass for suffix (generation)."""
        expert_outputs = self.lm_expert(
            inputs_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        hidden_states = getattr(expert_outputs, 'last_hidden_state', expert_outputs)
        past_kv = getattr(expert_outputs, 'past_key_values', None)
        
        return [hidden_states], past_kv