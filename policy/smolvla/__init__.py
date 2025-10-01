from .data_utils import SmolVLAProcess, SmolVLADataCollator
from .modeling import SmolVLAPolicy, SmolVLAPolicyConfig
from transformers import AutoTokenizer

def load_model(args):
    if args.is_pretrained:
        model = SmolVLAPolicy.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model.config.vlm_model_name)
        data_processor = SmolVLAProcess(tokenizer=tokenizer, max_length=model.config.tokenizer_max_length, padding=model.config.pad_language_to)
        data_collator = SmolVLADataCollator(max_state_dim=model.config.max_state_dim, max_action_dim=model.config.max_action_dim, resize_imgs_with_padding=model.config.resize_imgs_with_padding)
        model.data_processor = data_processor
        model.data_collator = data_collator
        model.tokenizer = tokenizer
    else:
        model_args = getattr(args, 'model_args', {})
        config = SmolVLAPolicyConfig(**model_args) 
        model = SmolVLAPolicy(config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_name)
    model.to('cuda')
    return {'model': model, 'tokenizer': tokenizer}

def get_data_processor(args, model_components):
    return SmolVLAProcess(tokenizer=model_components['tokenizer'], max_length=args.tokenizer_max_length, padding=args.pad_language_to)

def get_data_collator(args, model_components):
    return SmolVLADataCollator(max_state_dim=args.max_state_dim, max_action_dim=args.max_action_dim, resize_imgs_with_padding=args.resize_imgs_with_padding)


# def get_data_collator(args, model_components):
#     return Qwen2VLADataCollatorForSupervisedDataset(
#         multimodal_processor=model_components.get('multimodal_processor'),
#         computed_type=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
#         tokenizer=model_components.get('tokenizer'),
#         video=(args.history_images_length>=2)
#     )