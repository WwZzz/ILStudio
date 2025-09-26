# Configuration for SmolVLA adapted to IL-Studio
from dataclasses import dataclass, field
from transformers import PretrainedConfig
from typing import Dict, Any, List


@dataclass
class FeatureType:
    """Feature type constants."""
    VISUAL = "visual"
    STATE = "state"
    ACTION = "action"


@dataclass
class NormalizationMode:
    """Normalization mode constants."""
    IDENTITY = "identity"
    MEAN_STD = "mean_std"


@dataclass
class PolicyFeature:
    """Policy feature definition."""
    type: str
    shape: tuple

    def to_dict(self):
        return {"type": self.type, "shape": list(self.shape)}


class SmolVLAConfig(PretrainedConfig):
    """Configuration class for SmolVLA Policy adapted to IL-Studio."""
    
    def __init__(
        self,
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        
        # Normalization mapping
        normalization_mapping: Dict[str, str] = None,
        
        # Shorter state and action vectors will be padded
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        
        # Image preprocessing
        resize_imgs_with_padding: tuple = (512, 512),
        
        # Add empty images
        empty_cameras: int = 0,
        
        # Aloha adaptation
        adapt_to_pi_aloha: bool = False,
        use_delta_joint_actions_aloha: bool = False,
        
        # Tokenizer
        tokenizer_max_length: int = 48,
        
        # Decoding
        num_steps: int = 10,
        
        # Attention utils
        use_cache: bool = True,
        
        # Finetuning settings
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,
        
        # Training presets
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        
        scheduler_warmup_steps: int = 1000,
        scheduler_decay_steps: int = 30000,
        scheduler_decay_lr: float = 2.5e-6,
        
        # Model parameters
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = False,
        add_image_special_tokens: bool = False,
        attention_mode: str = "cross_attn",
        prefix_length: int = -1,
        pad_language_to: str = "longest",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        
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
        
        # Image preprocessing
        self.resize_imgs_with_padding = resize_imgs_with_padding
        
        # Empty cameras
        self.empty_cameras = empty_cameras
        
        # Aloha adaptation
        self.adapt_to_pi_aloha = adapt_to_pi_aloha
        self.use_delta_joint_actions_aloha = use_delta_joint_actions_aloha
        
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
        
        # Training presets
        self.optimizer_lr = optimizer_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_grad_clip_norm = optimizer_grad_clip_norm
        
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_steps = scheduler_decay_steps
        self.scheduler_decay_lr = scheduler_decay_lr
        
        # Model parameters
        self.vlm_model_name = vlm_model_name
        self.load_vlm_weights = load_vlm_weights
        self.add_image_special_tokens = add_image_special_tokens
        self.attention_mode = attention_mode
        self.prefix_length = prefix_length
        self.pad_language_to = pad_language_to
        self.num_expert_layers = num_expert_layers
        self.num_vlm_layers = num_vlm_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.expert_width_multiplier = expert_width_multiplier
        self.min_period = min_period
        self.max_period = max_period
        
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
    
    def to_dict(self) -> dict:
        """Make config JSON-serializable by converting PolicyFeature objects to plain dicts."""
        output = super().to_dict()
        def _serialize_features(feats):
            if not isinstance(feats, dict):
                return feats
            ser = {}
            for k, v in feats.items():
                if isinstance(v, PolicyFeature):
                    ser[k] = v.to_dict()
                else:
                    ser[k] = v
            return ser
        output["input_features"] = _serialize_features(getattr(self, "input_features", {}))
        output["output_features"] = _serialize_features(getattr(self, "output_features", {}))
        return output
    
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

