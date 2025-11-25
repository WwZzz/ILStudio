from .act import ACTPolicy, ACTPolicyConfig
from .data_utils import data_collator
import torch
from transformers import AutoConfig
from .trainer import Trainer

def load_model(args):
    if not args.is_training:
        model = ACTPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to('cuda')
    else:
        model_args = getattr(args, 'model_args', {})
        config = ACTPolicyConfig(**model_args) 
        model = ACTPolicy(config=config)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator