import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from collections import deque
from torch import Tensor
from typing import List
import safetensors
import logging
import torch
import numpy as np
import jax
import os
import shutil
import numpy as np
import sentencepiece
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download

class OpenPiPolicyConfig(PretrainedConfig):
    def __init__(self, 
            pytorch_training_precision: str='bfloat16',
            max_action_dim: int=32,
            action_dim: int=14,
            action_horizon: int=50,
            max_token_len: int=48,
            paligemma_variant: str="gemma_2b",
            action_expert_variant: str="gemma_300m",
            pi05: bool=False,
            discrete_state_input: bool=False,
            state_dim: int=32,
            pytorch_weight_path: str=None,
            lora_module: List[str] = ['gemma_expert', 'language_model'], # or ['gemma_expert']
            lora_r: int=16, # or 32
            lora_alpha: float=16.0, # or 32
            freeze_vision_tower: bool=False,
            **kwargs,
        ):
        self.pytorch_training_precision = pytorch_training_precision
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.pi05 = pi05
        self.discrete_state_input = discrete_state_input
        self.state_dim = state_dim
        self.pytorch_weight_path = pytorch_weight_path
        self.lora_module = lora_module
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.freeze_vision_tower = freeze_vision_tower

class OpenPiPolicy(PreTrainedModel):
    config_class = OpenPiPolicyConfig
    
    def __init__(self, config: OpenPiPolicyConfig):
        super().__init__(config)
        self.model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            max_token_len=config.max_token_len,
            paligemma_variant=getattr(config, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config, "pi05", False),
        )
        self.model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(self.model_cfg)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            enable_gradient_checkpointing = True
            self.model.gradient_checkpointing_enable()
        else:
            enable_gradient_checkpointing = False
        if config.pytorch_weight_path is not None:
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            safetensors.torch.load_model(
                (self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model), model_path
            )
            logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    
    def forward(self, observation, actions=None):
        if actions is not None:
            dev = next(self.model.parameters()).device
            observation = jax.tree.map(lambda x: x.to(dev), observation)
            actions = actions.to(dev)
            losses = self.model(observation, actions)
            return {'loss': losses.mean()}
        else:
            dev = next(self.model.parameters()).device
            observation = jax.tree.map(lambda x: x.to(dev), observation)
            action = self.model.sample_actions(dev, observation)
            return action
        
    def select_action(self, obs):
        num_obs = obs['state'].shape[0]
        instances = [{'state': obs['state'][i], 'image': obs['image'][i], 'raw_lang': obs['raw_lang'][i]} for i in range(num_obs)]
        processed_obs = [self.data_processor(instance) for instance in instances]
        batch_obs = self.data_collator(processed_obs)
        action = self.forward(**batch_obs)
        return action[:,:,:self.config.action_dim].cpu().numpy()
    
    def get_input_embeddings(self):
        return self.model.paligemma_with_expert.paligemma.language_model.embed_tokens
        
        
        