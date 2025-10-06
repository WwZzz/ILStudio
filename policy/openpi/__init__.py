from .modeling import OpenPiPolicyConfig, OpenPiPolicy
import openpi.models.tokenizer as _tokenizer
from .data_utils import OpenPiProcessor, OpenPiCollator
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, TaskType
import torch

def find_all_linear_names(model, lora_module=[]):
    cls = torch.nn.Linear
    lora_module_names = set()
    lora_keywords = lora_module
    for name, module in model.named_modules():
        if any([k in name for k in lora_module]) and isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def load_model(args):
    if args.is_pretrained:
        model = OpenPiPolicy.from_pretrained(args.model_name_or_path)
        model.data_processor = OpenPiProcessor(max_token_len=model.config.max_token_len, model_action_dim=model.config.max_action_dim, discrete_state_input=model.config.discrete_state_input)
        model.data_collator = OpenPiCollator()
    else:
        model_args = getattr(args, 'model_args', {})
        config = OpenPiPolicyConfig(**model_args) 
        model = OpenPiPolicy(config=config)
        if config.lora_module:
            modules_to_save_list = [n for n,p in model.named_parameters() if not n.startswith('model.paligemma_with_expert')]
            if not config.freeze_vision_tower:
                modules_to_save_list.append('model.paligemma_with_expert.paligemma.model.vision_tower')
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=find_all_linear_names(model, config.lora_module), 
                modules_to_save=modules_to_save_list,
            )
            model = get_peft_model(model, lora_config) 
            # set frozen parameters
            for n,p in model.named_parameters():
                if any([k in n for k in config.lora_module]) and 'lora' not in n:
                    p.requires_grad = False
                else:
                    if config.freeze_vision_tower and 'vision_tower' in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            model.print_trainable_parameters()
        
    model.to('cuda')
    return {'model': model}

def get_data_processor(args, model_components):
    return OpenPiProcessor(max_token_len=args.max_token_len, model_action_dim=args.max_action_dim, discrete_state_input=args.discrete_state_input)

def get_data_collator(args, model_components):
    return OpenPiCollator()


