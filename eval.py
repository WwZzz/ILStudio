import os
from torchvision import transforms
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed, load_normalizer_from_meta, load_data, WrappedDataset
import tensorflow as tf
# from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
from tqdm import tqdm
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
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
from configuration.constants import TASK_CONFIGS

e = IPython.embed
local_rank = None

@dataclass
class HyperArguments:
    # ############## model  ################
    is_pretrained: bool=field(default=True)
    device: str = 'cuda'
    
    # model_name: str = 'divla'
    # model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/divla_zscore"
    # norm_path: str = ''
    # chunk_size: int = field(default=16)
    # freq: int = 16
    # save_dir: str = 'results/divla_test'
    
    # model_name: str = 'diffusion_policy'
    # model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/dp_test/checkpoint-7700"
    # norm_path: str = ''
    # chunk_size: int = field(default=16)
    # freq: int = 16
    # save_dir: str = 'results/dp_test'

    model_name: str = 'diffusion_policy'
    model_name_or_path: str = "/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/diffusion_policy_transfer_cube_top_zscore_official_aug"
    norm_path: str = ''
    chunk_size: int = field(default=50)
    freq: int = 50
    save_dir: str = 'results/dp_aloha_transer-official-ema-freq50-dnoise10-aug'
    dataset_dir: str = '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_transfer_cube_scripted'
    
    ################ simulator #############
    # env_name: str = field(default='libero')
    # task: str = field(default="libero_object_1")
    # dataset_dir: str = '/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_object/h5v2'
    
    env_name: str = field(default='aloha')
    task: str = field(default="sim_transfer_cube_scripted")
    # dataset_dir: str = '/inspire/hdd/project/robot-action/public/data/act_aloha/sim_transfer_cube_human'
    fps: int = 50
    num_rollout: int = 4
    num_envs: int = 2
    max_timesteps: int = 400
    image_size: str = '(640, 480)' # (width, height)
    
    ctrl_space: str = 'joint'
    ctrl_type: str = 'abs'
    camera_ids: str = '[0]'
    
    #  ############ data ###################
    image_size_primary: str = "(640,480)"  # image size of non-wrist camera
    image_size_wrist: str = "(640,480)" # image size of wrist camera

  
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
    global local_rank
    # 用HFParser来传递参数，定义在上边的dataclass里
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    print(unknown_args)
    print(args)
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
    kwargs = norm_meta.get('kwargs', {'ctrl_type':'delta', 'ctrl_space':'ee'})
    return normalizers, kwargs['ctrl_space'], kwargs['ctrl_type'] 

if __name__=='__main__':
    set_seed(0)
    args = parse_param()
    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
    # load policy
    model_module = importlib.import_module(f"vla.{args.model_name}") 
    assert hasattr(model_module, 'load_model'), "model_name must provide API named `load_model` that returns dict like '\{'model':...\}'"
    model_components = model_module.load_model(args) # load_model是模型模块必须实现的接口
    model = model_components['model']
    policy = MetaPolicy(policy=model, freq=args.freq, action_normalizer=normalizers['action'], state_normalizer=normalizers['state'], ctrl_space=ctrl_space, ctrl_type=ctrl_type)
    # load env
    env_module = importlib.import_module(f"benchmark.{args.env_name}") 
    if not hasattr(env_module, 'create_env'): raise AttributeError(f"env {args.env_name} has no 'create_env'")
    def env_fn(config, env_handler):
        def create_env():
            return env_handler(config)
        return create_env

    all_eval_results = []
    num_iters = args.num_rollout//args.num_envs if args.num_rollout%args.num_envs==0 else args.num_rollout//args.num_envs+1
    for i in tqdm(range(args.num_rollout//args.num_envs), total=num_iters):
        num_envs = args.num_envs if i<num_iters-1 else args.num_rollout-i*args.num_envs
        # init video recorder
        if args.save_dir!='':
            video_dir = os.path.join(args.save_dir, args.env_name, 'video')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{args.task}_roll{i*args.num_envs}_{i*args.num_envs+num_envs}.mp4") 
            video_writer = imageio.get_writer(video_path, fps=args.fps)
        else:
            video_writer = None
        env_fns = [env_fn(args, env_module.create_env) for _ in range(num_envs)]
        env = SubprocVectorEnv(env_fns)
        # evaluate
        model.eval()
        eval_result = evaluate(args, policy, env, video_writer=video_writer)
        print(eval_result)
        all_eval_results.append(eval_result)
    
    eval_result = {
        'total_success': sum(eri['total_success'] for eri in all_eval_results),
        'total': sum(eri['total'] for eri in all_eval_results),
        'horizon': sum([eri['horizon'] for eri in all_eval_results], []),
        'horizon_success': sum([eri['horizon_success']*eri['total_success'] for eri in all_eval_results])
    }
    eval_result['success_rate'] = 1.0*eval_result['total_success']/eval_result['total']    
    eval_result['horizon_success']/=eval_result['total_success']
    # save result
    if args.save_dir!='':
        env_res_dir = os.path.join(args.save_dir, args.env_name)
        os.makedirs(env_res_dir, exist_ok=True)
        env_res_file = os.path.join(env_res_dir, f'{args.task}.json')
        # eval_result = {k:v.astype(np.float32) if isinstance(v, np.ndarray) else v for k,v in eval_result.items()}
        with open(env_res_file, 'w') as f:
            json.dump(eval_result, f)

    