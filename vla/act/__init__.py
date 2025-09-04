from .act import ACTPolicy, ACTPolicyConfig
from .data_utils import data_collator
import torch
from transformers import AutoConfig
from .trainer import Trainer

def load_model(args):
    use_bf16 = getattr(args, 'use_bf16', False)
    if args.is_pretrained:
        model = ACTPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.use_bf16 = use_bf16
        model.to('cuda')
    else:
        model_args = getattr(args, 'model_args', {})
        config = ACTPolicyConfig(camera_names=args.camera_names, history_len=1, state_dim = args.state_dim, action_dim=args.action_dim, prediction_len=args.chunk_size, chunk_size=args.chunk_size, num_queries=args.chunk_size, **model_args) 
        model = ACTPolicy(config=config, use_bf16=use_bf16)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator


