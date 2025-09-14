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
    """SmolVLM model with action expert."""
    
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
            
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.lm_expert = SmolVLMForConditionalGeneration(config=lm_expert_config)
        
        # Freeze vision encoder if needed
        if freeze_vision_encoder:
            for param in self.get_vlm_model().vision_model.parameters():
                param.requires_grad = False
                
        # Set training mode
        if train_expert_only:
            for param in self.get_vlm_model().parameters():
                param.requires_grad = False
            for param in self.lm_expert.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = True
                
        self.attention_mode = attention_mode
        self.self_attn_every_n_layers = self_attn_every_n_layers

    def get_vlm_model(self):
        """Get the VLM model."""
        return self.vlm

    def embed_image(self, images):
        """Embed images using the vision encoder."""
        return self.get_vlm_model().vision_model(images).last_hidden_state

    def embed_language_tokens(self, input_ids):
        """Embed language tokens."""
        return self.get_vlm_model().text_model.embed_tokens(input_ids)

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
        """Forward pass through the model."""
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
        
        # Process suffix through expert
        expert_outputs = self.lm_expert.text_model(
            inputs_embeds=suffix_embeds,
            attention_mask=attention_mask[:, prefix_embeds.shape[1]:],
            position_ids=position_ids[:, prefix_embeds.shape[1]:],
            use_cache=use_cache,
        )
        
        # Combine outputs
        combined_hidden_states = torch.cat([vlm_outputs.last_hidden_state, expert_outputs.last_hidden_state], dim=1)
        
        return [combined_hidden_states], expert_outputs.past_key_values

    def _forward_prefix(self, prefix_embeds, attention_mask, position_ids, use_cache, fill_kv_cache):
        """Forward pass for prefix (caching)."""
        vlm_outputs = self.get_vlm_model().text_model(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        
        if fill_kv_cache:
            return [vlm_outputs.last_hidden_state], vlm_outputs.past_key_values
        else:
            return [vlm_outputs.last_hidden_state], None

    def _forward_suffix(self, suffix_embeds, attention_mask, position_ids, past_key_values, use_cache):
        """Forward pass for suffix (generation)."""
        expert_outputs = self.lm_expert.text_model(
            inputs_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        return [expert_outputs.last_hidden_state], expert_outputs.past_key_values
