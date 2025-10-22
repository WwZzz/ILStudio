import numpy as np
from dataclasses import dataclass, field, fields, asdict
from torchvision import transforms
import pickle
import torch
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
# import tensorflow as tf
# from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio
import cv2


def get_images_from_metaobs(mobs): 
    images = mobs.image
    return [Image.fromarray(img.transpose(1,2,0)) for img in images]

def resize_with_pad(img, width, height, pad_value=-1, interpolation=cv2.INTER_LINEAR):
    """Resize (N,C,H,W) to (N,C,height,width) with aspect ratio preserved via padding.
    - Supports torch.Tensor or numpy.ndarray inputs
    - Pads on left and top (image aligned to bottom-right), matching existing behavior
    """
    is_torch = isinstance(img, torch.Tensor)
    device = img.device if is_torch else None
    dtype = img.dtype if is_torch else None
    arr = img.detach().cpu().numpy() if is_torch else img
    if arr.ndim != 4:
        raise ValueError(f"(n,c,h,w) expected, but {tuple(arr.shape)}")
    
    n, c, h, w = arr.shape
    if h==height and w==width: return img
    out_list = []
    for i in range(n):
        chw = arr[i]
        cur_h, cur_w = chw.shape[1], chw.shape[2]
        ratio = max(cur_w / width, cur_h / height)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)

        hwc = np.transpose(chw, (1, 2, 0))
        resized = cv2.resize(hwc, (resized_w, resized_h), interpolation=interpolation)

        canvas = np.full((height, width, c), pad_value, dtype=resized.dtype)
        y0 = height - resized_h
        x0 = width - resized_w
        canvas[y0:height, x0:width, :] = resized

        chw_out = np.transpose(canvas, (2, 0, 1))
        out_list.append(chw_out)

    out = np.stack(out_list, axis=0)
    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out.astype(arr.dtype, copy=False)

def organize_obs(obs: np.ndarray, ctrl_space='ee'):
    """Organize obs returned by SubprocVectorEnv into a dict"""
    # Lazy import to avoid circular dependency at module import time
    from .base import dict2meta
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
    res['state'] = res['state']
    # Note: camera selection is now handled in each environment's obs2meta method
    return dict2meta(res)

def evaluate(args, policy, env, video_writer=None):
    video_frames = [[] for _ in range(len(env))]
    horizons = np.ones(len(env))*args.max_timesteps
    # 开始测试
    with torch.inference_mode():
        time_start_eval = time.time()
        success =  np.zeros(len(env)).astype(np.bool8)
        obs = env.reset()
        obs = organize_obs(obs, args.ctrl_space)
        for t in range(args.max_timesteps):
            if video_writer is not None:
                frames = obs['image']
                if len(frames.shape)==5: frames = frames[:, 0]
                frames = frames.transpose(0, 2, 3, 1)
                for env_i in range(len(env)):
                    video_frames[env_i].append(frames[env_i])
            act = policy.select_action(obs, t)
            obs, reward, done, info = env.step(act)
            obs = organize_obs(obs, args.ctrl_space)
            # 判断是否成功
            success = success | done
            if success.all(): 
                for sidx in range(success.shape[0]):
                    if horizons[sidx]>t: horizons[sidx] = t
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
        'success': success.tolist(),
        'total_success': total_successes,
        'total': total,
        'success_rate': success_rate,
        'horizon': horizons.tolist(),
        'horizon_success': (success*horizons).sum()/(total_successes),
    }

def absolute_action_to_delta(maction, mobs):
    # Convert absolute action into relative action
    if maction.ctrl_type=='delta': return maction
    if maction.ctrl_space=='ee':
        maction.ctrl_type = 'delta'
        assert mobs is not None and mobs.state_ee is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.state_ee
    elif maction.ctrl_space=='joint':
        maction.ctrl_type = 'delta'
        assert mobs is not None and mobs.joint_state is not None, "failed to load state_ee from MetaObs"
        maction.action = maction.action - mobs.joint_state
    return maction