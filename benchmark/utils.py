import numpy as np
from dataclasses import dataclass, field, fields, asdict
from .base import MetaAction, MetaEnv, MetaObs, MetaPolicy
from torchvision import transforms
import pickle
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed
from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
import tensorflow as tf
from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio


def get_images_from_metaobs(mobs: MetaObs)-> List[Image]: 
    images = mobs.image
    return [Image.fromarray(img.transpose(1,2,0)) for img in img_array]

def evaluate(policy, env, query_frequency=16, max_timesteps=400, video_writer=None, time_wait=-1):
    assert raw_lang is not None, "raw lang is None"
    policy.policy.eval()
    # Normalizer
    normalizer = ... # process action or state
    # 设置动作队列
    action_queue = deque(maxlen=query_frequency)
    num_waits = 10
    video_frames = [[] for _ in range(len(env))]
    horizons = np.ones(len(env))*max_timesteps
    # 等待env初始化的时间
    if time_wait>0:
        for r in range(time_wait):
            # obs, reward, done, info = env.step([np.zeros((7,)).tolist() for _ in range(len(env))])
            obs, _, _, _ = env.step([np.zeros((7,)).tolist() for _ in range(len(env))])
    # 开始测试
    with torch.inference_mode():
        time_start_eval = time.time()
        success =  np.zeros(len(env)).astype(np.bool8)
        for t in range(max_timesteps):
            # 处理batch化的obs
            mobs = env.process_obs(obs)
            image_list.append(get_images_from_metaobs(mobs)) # image_list.append([Image.fromarray(img.squeeze(0).squeeze(0).transpose(1,2,0)) for img in img_array])
            if t % query_frequency == 0:
                # 反归一化state
                mobs = normalizer.denormalize(mobs)
                actions, outputs = policy.evaluate(mobs)
                mactions = policy.process_act(actions) # torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency]
                # clear previous actions
                while len(action_queue) > 0: action_queue.popleft()
                action_queue.extend(mactions)
            mact = action_queue.popleft()
            mact = normalizer.denormalize(mact)
            obs, reward, done, info = env.step(mact)
            success = success | done
            if success.all(): 
                break
            elif success.any():
                success_idx = np.where(success==True)[0]
                for sidx in success_idx: 
                    if horizons[sidx]>t: horizons[sidx] = t
            if video_writer is not None:
                frames = image_list[-1]
                for env_i in range(len(env)):
                    video_frames[env_i].append(np.array(frames[env_i]))
    env.close()
    total_successes = success.sum()
    total = len(env)
    success_rate = 1.0*total_successes/len(env)
    if video_writer is not None:
        for env_i in range(len(env)):
            for frame in video_frames[env_i]:
                video_writer.append_data(frame)
    return {
        'success': total_successes,
        'total': total,
        'success_rate': success_rate,
        'horizon': horizons,
    }

def absolute_action_to_relative(maction: MetaAction, mobs: MetaObs) -> MetaAction:
    # Convert absolute action into relative action
    if maction.is_delta: return maction
    if maction.atype=='ee':
        maction.is_delta = False
        assert mobs is not None and mobs.state_ee is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.state_ee
    elif maction.atype=='joint':
        maction.is_delta = False
        assert mobs is not None and mobs.joint_state is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.joint_state
    return maction