from .data_utils import SmolVLAProcess, SmolVLADataCollator
from .modeling import SmolVLAPolicy, SmolVLAPolicyConfig
from transformers import AutoTokenizer

def load_model(args):
    if args.is_pretrained:
        model = SmolVLAPolicy.from_pretrained(args.model_name_or_path)
    else:
        model_args = getattr(args, 'model_args', {})
        config = SmolVLAPolicyConfig() 
        model = SmolVLAPolicy(config=config)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_name)
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