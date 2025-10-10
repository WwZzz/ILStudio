from .modeling import OctoConfig, OctoPolicy
from .data_utils import OctoDataProcessor

def load_model(args):
    if args.is_pretrained:
        model = OctoPolicy.from_pretrained(args.model_name_or_path)
    else:
        model_args = getattr(args, 'model_args', {})
        config = OctoConfig(**model_args) 
        model = OctoPolicy(config=config)
    model.to('cuda')
    return {"model": model, 'text_processor': model.text_processor}

def get_data_processor(args, model_components):
    return OctoDataProcessor(model_components['text_processor'], model.config.use_wrist)
