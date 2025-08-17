import pickle
import os
import time
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
import importlib
import IPython
import torch
import vla.utils as ml_utils
from configuration.utils import *
from data_utils.utils import set_seed, WrappedDataset, load_data
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
from configuration.constants import TASK_CONFIGS

e = IPython.embed
local_rank = None

@dataclass
class HyperArguments(transformers.TrainingArguments):
    # ############## model  ################
    model_name: str = 'qwen2vl_dp'
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    is_pretrained: bool=field(default=False)
    output_dir:str = 'visualization/tmp'
    # ############# policy #################
    state_dim: int = 7
    action_dim: int = 7
    #  ############ data ###################
    action_normalize: str = 'minmax' # zscore, percentile
    state_normalize: str = 'minmax' # zscore, percentile
    chunk_size: int = field(default=16)
    image_size_primary: str = "(256,256)"  # image size of non-wrist camera
    image_size_wrist: str = "(256,256)" # image size of wrist camera
    use_reasoning: bool = False # whether to load reasoning data
    use_prev_subtask: bool = False # whether to add previous task into input
    
    lazy_preprocess: bool = False
    episode_first: bool = False  # batchsampler will samples episode index first and then samples timesteps
    select_seg_token_mask: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    task_name: str = field(default="stack_cube_2024_6_2") # task name corresponding to configuration/constants.py
    skip_mirrored_data: bool = field(default=False)
    preload_data: bool = field(default=False)

    history_images_length: int = 1 # length of history images
    #  ########### training ################
    using_ema: bool = field(default=False) # whether to use ema update whole module, default to false
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.95)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    remove_unused_columns: bool = field(default=False)
    flash_attn: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    resume_from_checkpoint: bool = field(default=False)
    llm_loss_weight: float = field(default=1.0)
    seed: int = field(default=0)
    # logger
    logging_dir: str = field(default='./logs')  # TensorBoard
    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=10)
    save_steps: int = field(default=10)
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=5000)
    # validate, unused
    do_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=200)
    per_device_eval_batch_size: int = field(default=32)
    load_pretrain: bool = False # loading pretrained VLA (For stage 3 training)
    dataloader_pin_memory: bool = False
    # lora, used when lora_enable is True
    use_quantization: bool=False
    lora_enable: bool = False # using lora or not
    lora_module: str = "vit,llm,merger" # which part to lora finetune, used when lora_enable is True
    lora_task_type: str = 'CAUSAL_LM'
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def _convert_to_type(value):
    """
    根据值的形式推断类型。支持 int, float 和 bool。
    """
    if not isinstance(value, str): return value
    # 尝试推断布尔值
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    # 尝试推断整型
    if value.isdigit():
        return int(value)
    # 尝试推断浮点数
    try:
        return float(value)
    except ValueError:
        pass
    # 否则，返回原始字符串
    return value

def parse_param():
    """
    Parse command line arguments and initialize configuration for model training.

    This function parses command line arguments into dataclass instances and sets up
    configuration for model training, including quantization settings and policy head
    configuration.

    Returns:
        args (HyperArguments): Training hyperparameters and settings

    Raises:
        NotImplementedError: If an unsupported policy head type is specified
    """
    global local_rank
    # 用HFParser来传递参数，定义在上边的dataclass里
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    local_rank = args.local_rank
    
    # 将unknown_args解析为字典
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]  # 去掉 '--' 前缀
            if "=" in key:  key, value = key.split('=', 1)
            else: value = True  # 如果没有指定值（如 --flag），默认设置为 True
            extra_args[key] = value
    model_args = {}
    # 动态将 `extra_args` 注入到 args 对象中
    for key, value in extra_args.items():
        try:
            value = _convert_to_type(value)
            if key.startswith('model.'): 
                model_args[key[6:]] = value # 动态获取自定义的model_args, i.e.，以model.为起始的字符串
            else:
                setattr(args, key, value) # 设置非模型相关的参数为args的属性
        except ValueError as e:
            print(f"Warning: {e}")
    args.model_args = model_args
    return args

def main(args):
    """
    Main training function for the VLA (Vision-Language-Action) model.

    Args:
        all_config (dict): Configuration dictionary containing:
            - model_args: Model architecture and loading arguments
            - data_args: Data processing and dataset arguments
            - training_args: Training hyperparameters and settings
            - action_head_args: Action head model configuration
        model_config (AutoConfig): Model configuration object for the Qwen2VLA model

    Returns:
        None. The trained model and statistics are saved to the output directory
        specified in training_args.
    """
    # 初始化任务信息
    set_seed(1)
    task_config = TASK_CONFIGS[args.task_name]
    args.camera_names = task_config['camera_names']
    args.image_sizes = [args.image_size_primary if 'primary' in cam else args.image_size_wrist for cam in args.camera_names]
    # 加载模型
    model_module = importlib.import_module(f"vla.{args.model_name}") 
    assert hasattr(model_module, 'load_model'), "model_name must provide API named `load_model` that returns dict like '\{'model':...\}'"
    model_components = model_module.load_model(args) # load_model是模型模块必须实现的接口
    model = model_components['model']
    ml_utils.print_model_trainable_information(model)
    # 加载数据集
    train_dataset, val_dataset = load_data(
        args, 
        task_config,
    )
    # 包装数据集
    get_data_processor = getattr(model_module, 'get_data_processor', None)
    data_module = dict(
        train_dataset=WrappedDataset(train_dataset, get_data_processor(train_dataset, args, model_components)) if get_data_processor is not None else train_dataset,
        eval_dataset=WrappedDataset(val_dataset, get_data_processor(val_dataset, args, model_components)) if get_data_processor is not None and val_dataset is not None else val_dataset,
        data_collator=model_module.get_data_collator(args, model_components) if hasattr(model_module, 'get_data_collator') else None,
    ) 
    # 获取 Trainer
    train_class = model_module.Trainer if hasattr(model_module, 'Trainer') else transformers.trainer.Trainer
    trainer = train_class(
        args=args,
        model=model,
        tokenizer=model_components.get('tokenizer', None),
        **data_module
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # 保存模型
    if trainer.is_world_process_zero():
        trainer.save_state()
        trainer.save_model(args.output_dir)

if __name__ == '__main__':
    args = parse_param()
    os.makedirs(args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(args.output_dir, ckpt_name) for ckpt_name in os.listdir(args.output_dir) if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(args.output_dir, ckpt_name))]
    if len(all_ckpts)==0: args.resume_from_checkpoint = False
    main(args)
