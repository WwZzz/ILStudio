from .diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyModel
from .utils import data_collator
import torch

def load_model(args):
    if not args.is_pretrained:
        config = DiffusionPolicyConfig(camera_names=args.camera_names, observation_horizon=1, action_dim=args.action_dim, state_dim = args.state_dim, prediction_horizon = args.chunk_size) 
        model = DiffusionPolicyModel(config=config)
    else:
        model = DiffusionPolicyModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator



