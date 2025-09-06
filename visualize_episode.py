import pickle
import os
import time
import cv2
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
import importlib
# import IPython  # Removed to avoid unnecessary dependency
import torch
import numpy as np
from configs.task.loader import load_task_config
from data_utils.utils import set_seed, WrappedDataset, load_data
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
# from data_utils.cooker.episode_tools import extract_traj_feature_of_episode
# e = IPython.embed  # Removed to avoid unnecessary dependency
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
    task_name: str = field(default="stack_cube_2024_6_2") # task name corresponding to configs/constants.py
    skip_mirrored_data: bool = field(default=False)


    history_images_length: int = 1 # length of history images
    #  ########### training ################
    using_ema: bool = field(default=False) # whether to use ema update whole module, default to false
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
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
def save_images_to_mp4(image_list, output_path, fps=30, codec='mp4v'):
    """
    将包含图像的 numpy 数组列表保存为 MP4 视频文件。

    参数：
    - image_list: 包含图像的 numpy 数组列表，每个数组的形状应为 (height, width, channels)。
    - output_path: 输出视频文件的路径，应以 .mp4 结尾。
    - fps: 视频的帧率，默认为 30。
    - codec: 视频编码器，默认为 'mp4v'，适用于 MP4 格式。
    """
    # 检查输入列表是否为空
    if image_list is None or len(image_list)==0:
        raise ValueError("输入的图像列表为空！")

    # 获取第一张图像的尺寸和通道数
    height, width, channels = image_list[0].shape

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 检查 VideoWriter 是否成功创建
    if not video_writer.isOpened():
        raise IOError(f"无法创建视频文件：{output_path}")

    # 遍历图像列表并写入视频
    for image in image_list:
        # 确保图像的形状与第一张图像一致
        if image.shape != (height, width, channels):
            raise ValueError("图像尺寸或通道数不一致！")
        if channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 写入帧
        video_writer.write(image)
    # 释放 VideoWriter 对象
    video_writer.release()
    print(f"视频已成功保存到：{output_path}")

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

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
    return args

def main(args):
    # 初始化任务信息
    set_seed(1)
    task_config = load_task_config(args.task_name)
    args.camera_names = task_config['camera_names']
    args.image_sizes = [args.image_size_primary if 'primary' in cam else args.image_size_wrist for cam in args.camera_names]

    # 加载数据集
    train_dataset, _ = load_data(
        args, 
        task_config,
    )

    all_feats = {}
    datasets = train_dataset.datasets
    data = datasets[0]
    for i in range(len(data)):
        images = data.extract_from_episode(i, ['image'])['image']
        for k in images.keys():
            save_images_to_mp4(images[k], os.path.join(args.output_dir, f'{i}_{k}.mp4'), fps=25)
        input("Enter to Continue")

        
if __name__ == '__main__':
    args = parse_param()
    os.makedirs(args.output_dir, exist_ok=True)
    all_ckpts = [os.path.join(args.output_dir, ckpt_name) for ckpt_name in os.listdir(args.output_dir) if ckpt_name.startswith('checkpoint-') and os.path.isdir(os.path.join(args.output_dir, ckpt_name))]
    if len(all_ckpts)==0: args.resume_from_checkpoint = False
    main(args)
