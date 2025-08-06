import numpy as np
from dataclasses import dataclass, field, fields, asdict
from .base import MetaAction, MetaEnv, MetaObs, MetaPolicy, dict2meta
from torchvision import transforms
import pickle
import torch
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed
import tensorflow as tf
from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio


def get_images_from_metaobs(mobs: MetaObs): 
    images = mobs.image
    return [Image.fromarray(img.transpose(1,2,0)) for img in img_array]

def organize_obs(obs: np.ndarray, space_name='ee', camera_ids=[0]):
    """Organize obs returned by SubprocVectorEnv into a dict"""
    if isinstance(obs, dict): return obs
    if isinstance(obs[0], dict):
        all_obs_dict = list(obs)
    else:
        all_obs_dict = list(asdict(obsi) for obsi in obs)
    assert len(all_obs_dict)>0, "emypt observation"
    all_keys = list(all_obs_dict[0].keys())
    res = {k:[vi[k] for vi in all_obs_dict] for k in all_keys}
    for k in res:
        if isinstance(res[k][0], np.ndarray):
            res[k] = np.stack(res[k])
        elif res[k][0] is None:
            res[k] = None
    res['state'] = res['state_ee'] if space_name=='ee' else res['state_joint']
    if isinstance(camera_ids, str): camera_ids = eval(camera_ids)
    res['image'] = res['image'][:,camera_ids]
    return dict2meta(res)

def evaluate(args, policy, env, video_writer=None):
    video_frames = [[] for _ in range(len(env))]
    horizons = np.ones(len(env))*args.max_timesteps
    # 开始测试
    with torch.inference_mode():
        time_start_eval = time.time()
        success =  np.zeros(len(env)).astype(np.bool8)
        obs = env.reset()
        obs = organize_obs(obs, args.space_name, args.camera_ids)
        for t in range(args.max_timesteps):
            if video_writer is not None:
                frames = obs['image']
                if len(frames.shape)==5: frames = frames[:, 0]
                frames = frames.transpose(0, 2, 3, 1)
                for env_i in range(len(env)):
                    video_frames[env_i].append(frames[env_i])
            act = policy.select_action(obs, t)
            obs, reward, done, info = env.step(act)
            obs = organize_obs(obs, args.space_name, args.camera_ids)
            # 判断是否成功
            success = success | done
            if success.all(): 
                break
            elif success.any():
                success_idx = np.where(success==True)[0]
                for sidx in success_idx: 
                    if horizons[sidx]>t: horizons[sidx] = t

    env.close()
    # 汇总结果
    total_successes = int(success.sum().item())
    total = len(env)
    success_rate = 1.0*total_successes/len(env)
    # 保存视频
    if video_writer is not None:
        for env_i in range(len(env)):
            for frame in video_frames[env_i]:
                video_writer.append_data(frame)
    return {
        'success': total_successes,
        'total': total,
        'success_rate': success_rate,
        'horizon': horizons.tolist(),
    }

def absolute_action_to_relative(maction: MetaAction, mobs: MetaObs) -> MetaAction:
    # Convert absolute action into relative action
    if maction.is_delta: return maction
    if maction.space_name=='ee':
        maction.is_delta = False
        assert mobs is not None and mobs.state_ee is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.state_ee
    elif maction.space_name=='joint':
        maction.is_delta = False
        assert mobs is not None and mobs.joint_state is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.joint_state
    return maction