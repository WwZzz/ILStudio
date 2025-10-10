from .cnnmlp import CNNMLPPolicy, CNNMLPPolicyConfig
from .data_utils import data_collator
import torch
from transformers import AutoConfig
from .trainer import Trainer

def load_model(args):
    if args.is_pretrained:
        model = CNNMLPPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to('cuda')
    else:
        model_args = getattr(args, 'model_args', {})
        config = CNNMLPPolicyConfig(camera_names=args.camera_names, state_dim = args.state_dim, action_dim=args.action_dim, chunk_size=args.chunk_size, **model_args) 
        model = CNNMLPPolicy(config=config)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator


