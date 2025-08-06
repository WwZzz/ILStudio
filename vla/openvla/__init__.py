import os
from pathlib import Path
from typing import Optional
import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
sys.path.append(os.path.dirname(__file__))
import warnings
warnings.filterwarnings("once", category=UserWarning)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from PIL import Image

IGNORE_INDEX=-100

        
class Processor:
    def __init__(self, action_tokenizer, base_tokenizer, image_transform):
        self.prompt_builder_fn = PurePromptBuilder
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
    
    def __call__(self, sample):
        image = Image.fromarray(sample['image'][0].permute(1,2,0).numpy())
        # image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = sample['action']
        if action.shape[0]>1:
            warnings.warn("Raw OpenVLA only supports actions without chunking")
        action = action[0]
        instruction = sample['raw_lang']
        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

def load_model(args):
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.use_quantization:
        assert args.lora_enable, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16, # bf16
        quantization_config=quantization_config if args.use_quantization else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    if args.lora_enable:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=min(args.lora_alpha, 16),
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",    # 所有 linear 层都添加 LoRA
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    return {'model':vla, 'processor':processor, 'action_tokenizer':action_tokenizer, 'tokenizer':processor.tokenizer}
    
# def wrap_data(dataset, args, model_components):
#     action_tokenizer = model_components['action_tokenizer']
#     base_tokenizer = model_components['tokenizer']
#     image_transform = model_components['processor'].image_processor.apply_transform
#     return WrappedDataset(dataset, action_tokenizer=action_tokenizer, base_tokenizer=base_tokenizer, image_transform=image_transform)

class MyCollator:
    def __init__(self, collator, dtype=torch.bfloat16):
        self.collator = collator
        self.dtype = dtype
    
    def __call__(self, *args, **kwargs):
        batch = self.collator(*args, **kwargs)
        batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
        return batch

def get_data_collator(args, model_components):
    tokenizer = model_components['tokenizer']
    collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right")
    return MyCollator(collator)

def get_data_processor(dataset, args, model_components):
    action_tokenizer = model_components['action_tokenizer']
    base_tokenizer = model_components['tokenizer']
    image_transform = model_components['processor'].image_processor.apply_transform
    return Processor(action_tokenizer=action_tokenizer, base_tokenizer=base_tokenizer, image_transform=image_transform)