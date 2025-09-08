from .modeling_pi0 import PI0FlowMatching, PI0FlowMatchingConfig
from transformers import AutoTokenizer
from .data_utils import DataCollator

def load_model(args):
    if args.is_pretrained:
        model = PI0FlowMatching.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        config = PI0FlowMatchingConfig(max_state_dim=args.state_dim, max_action_dim=args.action_dim, n_action_steps=args.chunk_size)
        model = PI0FlowMatching(config)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    return {'model':model, 'tokenizer':tokenizer}

# def wrap_data(dataset, args, model_components):
#     return dataset

def get_data_collator(args, model_components):
    config = model_components['model'].config
    return DataCollator(config, model_components['tokenizer'], config.tokenizer_max_length, config.resize_imgs_with_padding, config.image_features, config.empty_cameras)

