from transformers import PreTrainedModel
import torch.nn as nn
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from .detr_vae import build
from transformers import PretrainedConfig

class ACTPolicyConfig(PretrainedConfig):
    """
    Configuration class for ACTPolicy, inheriting from transformers' PretrainedConfig.
    This class includes all arguments defined in the provided argparse parser.
    """
    def __init__(
        self,
        # Training-related hyperparameters
        lr_backbone=1e-5,
        kl_weight = 10,
        # Model parameters
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        camera_names=['primary'],  # Expected to be a list
        enc_layers=4,
        dec_layers=7,
        dim_feedforward=3200,
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        chunk_size=400,
        pre_norm=False,
        masks=False,
        # Policy-specific arguments
        state_dim=14,
        action_dim=14,
        **kwargs,  # Allow for future extensions
    ):
        super().__init__(**kwargs)
        
        # Store all arguments as instance attributes
        self.lr_backbone = lr_backbone
        # Backbone settings
        self.backbone = backbone
        self.dilation = dilation
        self.position_embedding = position_embedding
        self.camera_names = camera_names if camera_names is not None else []
        # Transformer settings
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nheads = nheads
        self.num_queries = chunk_size
        self.chunk_size = chunk_size
        self.pre_norm = pre_norm
        self.masks = masks
        # Task-related arguments
        self.kl_weight = kl_weight
        self.state_dim = state_dim
        self.action_dim = action_dim
    

class ACTPolicy(PreTrainedModel):
    config_class = ACTPolicyConfig  # 配置类

    def __init__(self, config):
        super().__init__(config)
        # 构建模型和优化器
        self.model = build(config)
        self.kl_weight = config.kl_weight
        # print(f'KL Weight {self.kl_weight}')

    
    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        Forward method for training and inference. Trainer calls this method automatically.
        
        Args:
            qpos: Tensor, shape (batch_size, state_dim), robot state.
            image: Tensor, shape (batch_size, C, H, W), normalized visual inputs.
            actions: Tensor, shape (batch_size, num_queries, action_dim), action sequences for training.
            is_pad: Tensor, shape (batch_size, num_queries), padding mask.
        Returns:
            Loss dictionary during training; sampled actions during inference.
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def select_action(self, obs):
        # process data
        device = next(self.parameters()).device  # Get model's device
        obs = {k:torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k,v in obs.items()}
        obs['image'] = obs['image']/255.0
        # inference
        a_hat, _, (_, _) = self.model(obs['state'], obs['image'], None)
        return a_hat

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld