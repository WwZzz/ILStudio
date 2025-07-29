from .modeling import QwenVLPolicyConfig
from .modeling import QwenVLForPolicy
from .trainer import Trainer
from .data_utils import WrappedDataset, Qwen2VLAProcess, Qwen2VLADataCollatorForSupervisedDataset
import transformers
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model


def load_model(args):
    # 先加载config
    config = QwenVLPolicyConfig(vlm_model_name_or_path=args.model_name_or_path, policy_action_dim = args.action_dim, policy_state_dim = args.state_dim, policy_prediction_horizon = args.chunk_size) if not args.is_pretrained else AutoConfig.from_pretrained(args.model_name_or_path)
    config.llm_loss_weight = args.llm_loss_weight
    # 加载模型组件
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # load qwen2_vl tokenizer
    multimodal_processor = AutoProcessor.from_pretrained(args.model_name_or_path) # load qwen2_vl input processor
    # 加载model
    kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
    # 这里要获取model的类型，需要动态import
    if args.is_pretrained:
        model = QwenVLForPolicy.from_pretrained(args.model_name_or_path, trust_remote_code=True).to(torch.bfloat16)
    else:
        model = QwenVLForPolicy(config=config).to(torch.bfloat16)
    model.config.use_cache = False
    model.requires_grad_(not args.freeze_backbone)
    # 是否启用梯度检查点，这个就是反向传播时重新计算激活值，从而不保存激活值，省显存
    if args.gradient_checkpointing:
        if hasattr(model.vlm, "enable_input_require_grads"):
            model.vlm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.vlm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # 加载lora
    if args.lora_enable:
        # 加载Lora的参数
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model, print, args.lora_module), # 默认只有vit
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=args.lora_task_type,
        )
        if args.bits == 16:
            if args.bf16: model.to(torch.bfloat16)
            if args.fp16: model.to(torch.float16)
        model = get_peft_model(model, lora_config) # !!!only set lora weights to requires_grad True!!!
        model.print_trainable_parameters()
    else:
        if hasattr(model, 'set_requires_grad'):
            model.set_requires_grad(args)
        else:
            warnings.warn("Failed to set requires_grad for modules because `set_requires_grad` method doesn't exist")
    # 模型放到device上
    vision_tower = model.vlm.visual
    vision_tower.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.device)
    model.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.device)
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    # 设置config里头的自定义参数
    model.config.non_lora_lr = args.non_lora_lr
    model.config.use_cache = True
    model.config.save_pretrained(args.output_dir)
    return {
        'model': model, 
        'tokenizer': tokenizer,
        'multimodal_processor': multimodal_processor,
    }


def wrap_data(dataset, args, model_components):
    processor = Qwen2VLAProcess(tokenizer=model_components['tokenizer'], multimodal_processor=model_components['multimodal_processor'], camera_names=dataset.camera_names)
    return WrappedDataset(dataset, processor)

def get_data_collator(args, model_components):
    return Qwen2VLADataCollatorForSupervisedDataset(
        multimodal_processor=model_components.get('multimodal_processor'),
        computed_type=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        tokenizer=model_components.get('tokenizer'),
        video=(args.history_images_length>=2)
    )
