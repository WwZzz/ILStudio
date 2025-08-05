from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
from torchvision import transforms
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed, load_normalizer_from_meta
import tensorflow as tf
from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio
from benchmark.utils import evaluate
from benchmark.base import MetaPolicy
import pickle
import transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"
import importlib
import IPython
import torch
import vla.utils as ml_utils
from configuration.utils import *
from data_utils.utils import load_data, set_seed, WrappedDataset
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
from configuration.constants import TASK_CONFIGS

e = IPython.embed
local_rank = None

@dataclass
class HyperArguments:
    # ############## model  ################
    model_name: str = 'qwen2vl_dp'
    model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/qdp_zscore_pipe/checkpoint-40"
    is_pretrained: bool=field(default=True)
    ################ simulator #############
    env_name: str = field(default='libero')
    task: str = field(default="libero_object_0")
    num_rollout: int = 10
    num_envs: int = 5
    max_timesteps: int = 400
    freq: int = 10
    image_size: str = '(256, 256)'
    norm_path: str = ''
    dataset_dir: str = '/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_object/h5v2'
    save_dir: str = 'tmp_dp-zscore50000'
    space_name: str = 'ee'
    abs_control: bool = False
    camera_ids: str = '[0]'
    
    #  ############ data ###################
    chunk_size: int = field(default=16)
    image_size_primary: str = "(256,256)"  # image size of non-wrist camera
    image_size_wrist: str = "(256,256)" # image size of wrist camera
    use_reasoning: bool = False # whether to load reasoning data
    use_prev_subtask: bool = False # whether to add previous task into input
    abs_control: bool = False
    fps: int = 20
    
    # lora, used when lora_enable is True
    lora_enable: bool = True # using lora or not
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

def parse_param():
    global local_rank
    # 用HFParser来传递参数，定义在上边的dataclass里
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return args

def load_normalizers(args):
    # load normalizers
    if args.norm_path=='':
        res = os.path.join(os.path.dirname(args.model_name_or_path), 'normalize.json')
        if not os.path.exists(res):
            res = os.path.join(args.model_name_or_path, 'normalize.json')
            if not os.path.exists(res):
                raise FileNotFoundError("No normalize.json found")
    else:
        res = args.norm_path
    with open(res, 'r') as f:
        norm_meta = json.load(f)
    normalizers = load_normalizer_from_meta(args.dataset_dir, norm_meta)
    return normalizers

if __name__=='__main__':
    set_seed(0)
    args = parse_param()
    normalizers = load_normalizers(args)
    # load policy
    model_module = importlib.import_module(f"vla.{args.model_name}") 
    assert hasattr(model_module, 'load_model'), "model_name must provide API named `load_model` that returns dict like '\{'model':...\}'"
    model_components = model_module.load_model(args) # load_model是模型模块必须实现的接口
    model = model_components['model']
    policy = MetaPolicy(policy=model, freq=args.freq, action_normalizer=normalizers['action'], state_normalizer=normalizers['state'], ctrl_space=args.space_name, abs_ctrl=args.abs_control)
    # load env
    env_module = importlib.import_module(f"benchmark.{args.env_name}") 
    if not hasattr(env_module, 'create_env'): raise AttributeError(f"env {args.env_name} has no 'create_env'")
    def env_fn(config, env_handler):
        def create_env():
            return env_handler(config)
        return create_env
    # env = env_module.create_env(args)
    env_fns = [env_fn(args, env_module.create_env) for _ in range(args.num_envs)]
    env = SubprocVectorEnv(env_fns)
    
    # init video writer
    if args.save_dir!='':
        video_dir = os.path.join(args.save_dir, args.env_name, 'video')
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{args.task}.mp4") 
        video_writer = imageio.get_writer(video_path, fps=args.fps)
    else:
        video_writer = None
    # evaluate
    model.eval()
    eval_result = evaluate(args, policy, env, video_writer=video_writer)
    print(eval_result)
    # save result
    if args.save_dir!='':
        env_res_dir = os.path.join(args.save_dir, args.env_name)
        os.makedirs(env_res_dir, exist_ok=True)
        env_res_file = os.path.join(env_res_dir, f'{args.task}.json')
        with open(env_res_file, 'w') as f:
            json.dump(eval_result, f)

    