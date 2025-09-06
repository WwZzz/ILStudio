import os
from torchvision import transforms
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed, _convert_to_type, load_normalizers
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
os.environ['DEVICE'] = "cuda"
import importlib
# import IPython  # Removed to avoid unnecessary dependency
import torch
from configs.task.loader import load_task_config
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List
import multiprocessing as mp
# e = IPython.embed  # Removed to avoid unnecessary dependency
local_rank = None

# Removed HyperArguments dataclass - using simple argparse instead

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model')
    
    # Model arguments
    parser.add_argument('--is_pretrained', action='store_true', default=True,
                       help='Whether to use pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    # Policy config system
    parser.add_argument('--policy_config', type=str, default='configs/policy/diffusion_policy.yaml',
                       help='Policy config file path')
    parser.add_argument('--model_name_or_path', type=str, 
                       default='/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/diffusion_policy_transfer_cube_top_zscore_official_aug',
                       help='Path to the model checkpoint')
    parser.add_argument('--norm_path', type=str, default='',
                       help='Path to normalization data')
    parser.add_argument('--save_dir', type=str, default='results/dp_aloha_transer-official-ema-freq50-dnoise10-aug',
                       help='Directory to save results')
    parser.add_argument('--dataset_dir', type=str, default='',
                       help='Dataset directory')
    
    # Simulator arguments
    parser.add_argument('--env_name', type=str, default='aloha',
                       help='Environment name')
    parser.add_argument('--task', type=str, default='sim_transfer_cube_scripted',
                       help='Task name')
    parser.add_argument('--fps', type=int, default=50,
                       help='Frames per second')
    parser.add_argument('--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('--num_envs', type=int, default=2,
                       help='Number of environments')
    parser.add_argument('--max_timesteps', type=int, default=400,
                       help='Maximum timesteps per episode')
    parser.add_argument('--image_size', type=str, default='(640, 480)',
                       help='Image size (width, height)')
    parser.add_argument('--ctrl_space', type=str, default='joint',
                       help='Control space')
    parser.add_argument('--ctrl_type', type=str, default='abs',
                       help='Control type')
    parser.add_argument('--camera_ids', type=str, default='[0]',
                       help='Camera IDs')
    parser.add_argument('--use_spawn', action='store_true',
                       help='Use spawn method for multiprocessing')
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    set_seed(0)
    args = parse_param()
    if args.use_spawn: mp.set_start_method('spawn', force=True)

    # For evaluation, parameters will be loaded from saved model config
    # No need to load task config parameters

    normalizers, ctrl_space, ctrl_type = load_normalizers(args)
    args.ctrl_space, args.ctrl_type = ctrl_space, ctrl_type
    
    # Load policy using policy config system for evaluation - uses saved model config
    print(f"Loading policy config: {args.policy_config}")
    from policy.policy_loader import load_policy_model_for_evaluation
    model_components = load_policy_model_for_evaluation(args.policy_config, args)
    model = model_components['model']
    config = model_components.get('config', None)
    if config:
        print(f"Loaded config from YAML: {type(config).__name__}")
    policy = MetaPolicy(policy=model, chunk_size=args.chunk_size, action_normalizer=normalizers['action'], state_normalizer=normalizers['state'], ctrl_space=ctrl_space, ctrl_type=ctrl_type)
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

    