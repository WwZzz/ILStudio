import torch
import transformers
import copy
from dataclasses import dataclass, field, fields, asdict
from peft import LoraConfig, get_peft_model
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
# from policy.qwen2vl_dp.modeling import QwenVLForPolicy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor
import warnings
import os

def find_all_linear_names(model, rank0_print, lora_module=None):
    cls = torch.nn.Linear
    lora_module_names = set()
    # 指定不使用lora的算子关键字，连接V和L的Connector，lm_head, 动作proj，reasoning proj, film，merger，这些都不用lora
    multimodal_keywords = ['multi_modal_projector', 'lm_head', 'input_action_proj', 'reasoning_action_proj', 'reasoning_film', 'merger']
    if 'vit' not in lora_module:
        multimodal_keywords.append("vision_tower")
    if 'llm' not in lora_module:
        multimodal_keywords.append("language_model")
    if 'di_head' not in lora_module: # not lora finetune policy_head
        multimodal_keywords.append("policy_head")
    rank0_print("##" * 20)
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords): continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_model_trainable_information(model, rank0_print=None):
    if rank0_print is None: rank0_print = print
    lora_para = sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad and 'lora' in n))
    all_para = sum(p.numel() for n, p in model.named_parameters())
    train_para = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    rank0_print(f"Lora parameters/trainalbe parameters/all parameters:{lora_para/1000000}M/{train_para/1000000}M/{(all_para-lora_para)/1000000}M")

# 这个函数只加载模型本身，不加载checkpoint权重
def load_model(config=None, qwen2_vla_config=None, rank0_print=print, tokenizer=None):
    # config是外部参数，qwen2_vla_config是AutoConfig的参数
    model_args = config['model_args']
    training_args = config['training_args']
    data_args = config['data_args']
    action_args = config['action_head_args']
    kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16}
    # 这里要获取model的类型，需要动态import
    if model_args.is_pretrained:
        model = QwenVLForPolicy.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).to(torch.bfloat16)
    else:
        model = QwenVLForPolicy(config=qwen2_vla_config).to(torch.bfloat16)
    model.config.use_cache = False
    model.requires_grad_(not training_args.freeze_backbone)
    # 是否启用梯度检查点，这个就是反向传播时重新计算激活值，从而不保存激活值，省显存
    if training_args.gradient_checkpointing:
        if hasattr(model.vlm, "enable_input_require_grads"):
            model.vlm.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.vlm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # 加载lora
    if training_args.lora_enable:
        # 加载Lora的参数
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, rank0_print, training_args.lora_module), # 默认只有vit
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=training_args.lora_task_type,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("##" * 20)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config) # !!!only set lora weights to requires_grad True!!!
        model.print_trainable_parameters()
    else:
        if hasattr(model, 'set_requires_grad'):
            model.set_requires_grad(training_args)
        else:
            warnings.warn("Failed to set requires_grad for modules because `set_requires_grad` method doesn't exist")

    # 模型放到device上
    vision_tower = model.vlm.visual
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # 设置config里头的自定义参数
    model.config.non_lora_lr = training_args.non_lora_lr
    print_model_trainable_information(model, rank0_print=rank0_print)
    return model

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_merge_lora_weights(model_path=None, model_base=None, kwargs=None):
    path = model_path.split('/')[0:-1]
    root_path = '/'.join(path)
    lora_cfg_pretrained = AutoConfig.from_pretrained(root_path)
    # config = lora_cfg_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)  # default use_fast=False
    print('Loading QWen2-VLA from base model...')
    model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                 config=lora_cfg_pretrained, **kwargs)

    print('Loading additional QWen2-VLA weights expecially non-lora part(diffusion head)...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                         )
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')

        non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                           non_lora_trainables.items()}
    if any(k.startswith('model.policy_head.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                               non_lora_trainables.items()}

    # Delete the parameters related to Lora
    keys_to_del = []
    for k, v in non_lora_trainables.items():
        if 'lora' in k:
            keys_to_del.append(k)
    for key in keys_to_del:
        del non_lora_trainables[key]

    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    return model, tokenizer

def load_model_for_eval(model_path, model_base, load_8bit=False, load_4bit=False, device_map="cuda", policy_config=None):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16
    
    if 'qwen2' in model_path.lower():

        if 'lora' in model_path.lower() and model_base is None: # only for lora finetuning
            warnings.warn(
                'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_path.lower() and model_base is not None: # only for lora finetuning
            if policy_config['pretrain_path'] is not None:
                ps = model_path.split('/')
                if not os.path.exists(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights')):
                    print("merging pretrained weights.......")
                    model, tokenizer = load_merge_lora_weights(model_path=policy_config['pretrain_path'], model_base=model_base, kwargs=kwargs)

                    os.makedirs(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'), exist_ok=True)
                    model.save_pretrained(
                        os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'))
                    tokenizer.save_pretrained(os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'))

                print("loading pretrained weights as base model.......")
                model, tokenizer = load_merge_lora_weights(model_path=model_path, model_base=os.path.join(policy_config['pretrain_path'], 'pretrain_merge_weights'), kwargs=kwargs)

            else:
                model, tokenizer = load_merge_lora_weights(model_path=model_path, model_base=model_base, kwargs=kwargs)


            # model = model.to(torch.bfloat16)
        elif model_base is not None:
            # this may be mm projector only
            print('Loading QWen2-VLA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                         **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print("load QWen2-VLA!!!")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                use_safetensors=True,
                **kwargs).to("cuda")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                         device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.bfloat16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


    multi_modal_processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    model.to(device="cuda")
    print(kwargs)
    # print(model)
    return tokenizer, model, multi_modal_processor, context_len

