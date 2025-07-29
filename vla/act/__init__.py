from .act import ACTPolicy, ACTPolicyConfig
from .data_utils import data_collator
import torch

def load_model(args):
    config = ACTPolicyConfig(camera_names=args.camera_names, history_len=1, action_dim=args.action_dim, state_dim = args.state_dim, prediction_len=args.chunk_size, chunk_size=args.chunk_size, num_queries=args.chunk_size) if not args.is_pretrained else AutoConfig.from_pretrained(args.model_name_or_path)
    if args.is_pretrained:
        model = ACTPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        model = ACTPolicy(config=config)
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator



