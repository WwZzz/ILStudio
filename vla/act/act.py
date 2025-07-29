from transformers import PreTrainedModel
import torch.nn as nn
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
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        batch_size=2,
        epochs=300,
        lr_drop=200,
        
        # Model parameters
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        camera_names=['primary'],  # Expected to be a list
        enc_layers=4,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=400,
        pre_norm=False,
        masks=False,

        # Policy-specific arguments
        ckpt_dir=None,
        policy_class=None,
        task_name=None,
        seed=None,
        num_steps=None,
        kl_weight=0,
        chunk_size=None,
        temporal_agg=False,
        use_vq=False,
        vq_class=None,
        vq_dim=None,
        load_pretrain=False,
        action_dim=None,
        state_dim=None,
        eval_every=500,
        validate_every=500,
        save_every=500,
        resume_ckpt_path=None,
        no_encoder=False,
        skip_mirrored_data=False,
        actuator_network_dir=None,
        history_len=None,
        future_len=None,
        prediction_len=None,

        **kwargs,  # Allow for future extensions
    ):
        super().__init__(**kwargs)
        
        # Store all arguments as instance attributes
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.clip_max_norm = clip_max_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_drop = lr_drop

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
        self.num_queries = num_queries
        self.pre_norm = pre_norm
        self.masks = masks

        # Task-related arguments
        self.ckpt_dir = ckpt_dir
        self.policy_class = policy_class
        self.task_name = task_name
        self.seed = seed
        self.num_steps = num_steps
        self.kl_weight = kl_weight
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.vq = use_vq
        self.use_vq = use_vq
        self.vq_class = vq_class
        self.vq_dim = vq_dim
        self.load_pretrain = load_pretrain
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.eval_every = eval_every
        self.validate_every = validate_every
        self.save_every = save_every
        self.resume_ckpt_path = resume_ckpt_path
        self.no_encoder = no_encoder
        self.skip_mirrored_data = skip_mirrored_data
        self.actuator_network_dir = actuator_network_dir
        self.history_len = history_len
        self.future_len = future_len
        self.prediction_len = prediction_len

class ACTPolicy(PreTrainedModel):
    config_class = ACTPolicyConfig  # 配置类

    def __init__(self, config):
        super().__init__(config)
        # 构建模型和优化器
        self.model = build(config)
        self.kl_weight = config.kl_weight
        self.vq = config.vq
        # print(f'KL Weight {self.kl_weight}')

    def forward(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        """
        Forward method for training and inference. Trainer calls this method automatically.
        
        Args:
            qpos: Tensor, shape (batch_size, state_dim), robot state.
            image: Tensor, shape (batch_size, C, H, W), normalized visual inputs.
            actions: Tensor, shape (batch_size, num_queries, action_dim), action sequences for training.
            is_pad: Tensor, shape (batch_size, num_queries), padding mask.
            vq_sample: VQ-VAE sampling, optional tensor.
        Returns:
            Loss dictionary during training; sampled actions during inference.
        """
        env_state = None  # Assuming no environment state
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # Training mode
            # Trim actions and padding mask to num_queries
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Model forward propagation
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(
                qpos, image, env_state, actions, is_pad, vq_sample
            )

            # Calculate KL divergence loss
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            # VQ discrepancy loss
            loss_dict = dict()
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')

            # L1 action loss
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1

            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict

        else:  # Inference mode
            # Sample actions from the prior (no actions provided)
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample)
            return a_hat

    # def configure_optimizers(self):
    #     """ Return optimizer for trainer. """
    #     return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        """ Handle VQ encoding step. """
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)
        return binaries

    def save_pretrained(self, save_directory, state_dict=None, *args, **kwargs):
        """ Save model and configuration during Trainer execution. """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        self.config.save_pretrained(save_directory)

        # Save state_dict (weights)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """ Load model and configuration for Trainer execution. """


        # Load configuration
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Initialize model using the configuration
        model = cls(config)

        # Load weights
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        return model
    
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