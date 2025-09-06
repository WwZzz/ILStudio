from .modeling import QwenVLPolicyConfig
from .modeling import QwenVLForPolicy
from .trainer import Trainer
from .data_utils import Qwen2VLAProcess, Qwen2VLADataCollatorForSupervisedDataset
import transformers
from safetensors.torch import save_file, load_file
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import os

def find_all_linear_names(model, lora_module=None):
    cls = torch.nn.Linear
    lora_module_names = set()
    no_lora_keywords = ['multi_modal_projector', 'lm_head', 'policy_head']
    if 'vit' not in lora_module:
        no_lora_keywords.append("vision_tower")
    if 'llm' not in lora_module:
        no_lora_keywords.append("language_model")
    if 'merger' not in lora_module:
        no_lora_keywords.append('merger')
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in no_lora_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

def load_model(args):
    # 先加载config
    if args.is_pretrained: # 测试时加载
        config = QwenVLPolicyConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name_or_path)
        multimodal_processor = AutoProcessor.from_pretrained(config.vlm_model_name_or_path)
        peft_path = os.path.join(args.model_name_or_path, 'adapter_config.json')
        # 检测是否是peft
        if os.path.exists(peft_path):  
            model = QwenVLForPolicy(config=config)
            model = PeftModel.from_pretrained(model,args.model_name_or_path)
            extra_path = os.path.join(args.model_name_or_path, 'extra_trainable.safetensors')
            if os.path.exists(extra_path):
                extra_state = load_file(extra_path, device="cpu")
                missing, unexpected = model.load_state_dict(extra_state, strict=False)
            model = model.merge_and_unload().to(torch.bfloat16)
        else:
            model = QwenVLForPolicy.from_pretrained(args.model_name_or_path, config=config).to(torch.bfloat16)
    else: # 训练时加载
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # load qwen2_vl tokenizer
        multimodal_processor = AutoProcessor.from_pretrained(args.model_name_or_path) # load qwen2_vl input processor
        config = QwenVLPolicyConfig(vlm_model_name_or_path=args.model_name_or_path, policy_action_dim = args.action_dim, policy_state_dim = args.state_dim, policy_prediction_horizon = args.chunk_size) 
        # config.llm_loss_weight = args.llm_loss_weight
        model = QwenVLForPolicy(config=config).to(torch.bfloat16)
        # model.requires_grad_(not args.freeze_backbone)
        # 加载lora
        if args.lora_enable:
            # 加载Lora的参数
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=find_all_linear_names(model, args.lora_module), # 默认只有vit
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type=args.lora_task_type,
            )
            if args.bits == 16:
                if args.bf16: model.to(torch.bfloat16)
                if args.fp16: model.to(torch.float16)
            model = get_peft_model(model, lora_config) # !!!only set lora weights to requires_grad True!!!
            model.print_trainable_parameters()
        
    # 这里要获取model的类型，需要动态import
    model.tokenizer = tokenizer
    model.multimodal_processor = multimodal_processor
    model.config.use_cache = False
    # # 是否启用梯度检查点，反向传播时重新计算激活值，从而不保存激活值，省显存
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        if hasattr(model.vlm, "enable_input_require_grads"):
            model.vlm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.vlm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if hasattr(args, 'bf16'): # 训练时设置需要梯度的参数
        model.set_requires_grad(args)
        # 模型放到device上
        vision_tower = model.vlm.visual
        vision_tower.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.device)
        model.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.device)
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        
        # 设置config里头的自定义参数
        model.config.lora_lr = args.lora_lr
        model.config.use_cache = True
        model.config.save_pretrained(args.output_dir)
    else:
        model.to(dtype=torch.bfloat16, device=args.device)
    return {
        'model': model, 
        'tokenizer': tokenizer,
        'multimodal_processor': multimodal_processor,
    }


# def wrap_data(dataset, args, model_components):
#     processor = Qwen2VLAProcess(tokenizer=model_components['tokenizer'], multimodal_processor=model_components['multimodal_processor'], camera_names=dataset.camera_names)
#     return WrappedDataset(dataset, processor)

def get_data_processor(dataset, args, model_components):
    return Qwen2VLAProcess(tokenizer=model_components['tokenizer'], multimodal_processor=model_components['multimodal_processor'], camera_names=dataset.camera_names)

def get_data_collator(args, model_components):
    return Qwen2VLADataCollatorForSupervisedDataset(
        multimodal_processor=model_components.get('multimodal_processor'),
        computed_type=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        tokenizer=model_components.get('tokenizer'),
        video=(args.history_images_length>=2)
    )
