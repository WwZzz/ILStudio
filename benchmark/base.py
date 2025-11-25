import numpy as np
import cv2
from dataclasses import dataclass, field, fields, asdict
from collections import deque
from typing import Optional
import torch
from .utils import resize_with_pad

@dataclass
class MetaAction:
    ctrl_space: str = 'ee' # or 'joint'
    ctrl_type: str = 'delta' # absolute, relative, or delta control
    action: np.ndarray = None # action[-1] is gripper control signal, i.e., 1 is open and 0 is close
    gripper_continuous: bool = False # is gripper controlled by continuous action, where action[-1] is position ratio to the gripper width
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __len__(self):
        if self.action is None: return 0
        if len(self.action.shape)==1: return 1
        return self.action.shape[0]
    
    def to_batch(self):
        if self.action is not None: 
            self.action = self.action[np.newaxis, :]

    
@dataclass
class MetaObs:
    state: np.ndarray = None
    state_ee : np.ndarray = None
    state_joint: np.ndarray = None
    state_obj: np.ndarray = None
    image: np.ndarray = None # (K, C, H, W)
    depth: np.ndarray = None # (K, H, W)
    pc: np.array = None # (n, 3)
    raw_lang: str = ''
    timestep: int = -1
    
    def __getitem__(self, key):
        return getattr(self, key)

    def to_batch(self):
        all_keys = ['state', 'state_ee', 'state_joint', 'state_obj', 'image', 'depth', 'pc', 'timestep']
        for k in all_keys:
            if self[k] is not None and isinstance(self[k], np.ndarray):
                setattr(self, k, self[k][np.newaxis, :])
    
META_OBS_KEYS = [f.name for f in fields(MetaObs)]
META_ACT_KEYS = [f.name for f in fields(MetaAction)]

def dict2meta(data_dict, mtype='obs'):
    if mtype=='obs':
        new_dict = {k:v for k,v in data_dict.items() if k in META_OBS_KEYS}
        return MetaObs(**new_dict)
    else:
        new_dict = {k:v for k,v in data_dict.items() if k in META_ACT_KEYS}
        return MetaAction(**new_dict)

class MetaEnv:
    def __init__(self, env):
        self.env = env
        self.prev_obs = None
        
    def step(self, *args, **kwargs):
        act = self.meta2act(*args, **kwargs)
        results = list(self.env.step(act))
        self.prev_obs = results[0] = self.obs2meta(results[0])
        if isinstance(results[0], MetaObs): results[0] = asdict(results[0])
        return tuple(results)
    
    def obs2meta(self, raw_obs):
        # convert raw_obs into MetaObs WITHOUT normalizing data
        raise NotImplementedError
    
    def meta2act(self, action, *args):
        # convert MetaAction into env-specific action WITHOUT normalizing data
        raise NotImplementedError
    
    def reset(self):
        init_obs = self.env.reset()
        self.prev_obs = self.obs2meta(init_obs)
        return self.prev_obs
    
    def close(self):
        self.env.close()
    
class MetaPolicy:
    def __init__(self, policy, chunk_size:Optional[int] = None, action_normalizer=None, state_normalizer=None, ctrl_space='ee', ctrl_type='delta', img_size=None):
        self.policy = policy
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.action_queue = deque(maxlen=chunk_size) if (chunk_size is not None and chunk_size>0) else deque(maxlen=1000)
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        if img_size is not None:
            img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size

    def meta2obs(self, mobs: MetaObs):
        # convert MetaObs into policy-specific obs
        if hasattr(self.policy, 'meta2obs'):
            obs = self.policy.meta2obs(mobs)
        else:
            obs = asdict(mobs)
        return obs
    
    def act2meta(self, action, ctrl_space:str='ee', ctrl_type:str=True):
        # convert action into MetaAction, np.ndarray((chunk_size, action_dim), dtype=np.float32) as default
        if hasattr(self.policy, 'act2meta'):
            mact = self.policy.act2meta(action, ctrl_space=ctrl_space, ctrl_type=ctrl_type)
        else:
            if isinstance(action, torch.Tensor): action = action.float().cpu().numpy()
            mact = MetaAction(action=action, ctrl_space=ctrl_space, ctrl_type=ctrl_type) # (B, chunk_size, dim) or (chunk_size, dim)
        return mact 
    
    def is_action_queue_empty(self):
        return len(self.action_queue)==0

    def inference(self, mobs: MetaObs):
        normed_mobs = self.state_normalizer.normalize_metaobs(mobs, self.ctrl_space)
        # try resize image
        if self.img_size is not None:
            images = normed_mobs.image 
            if images.ndim == 5:
                images = images[:, -1, :, :, :].transpose(0, 2, 3, 1) # n, c, h,w -> n, h, w, c
                original_dim = 5
            images = [cv2.resize(img, (self.img_size[1], self.img_size[0])) for img in images] # cv2.resize receive (W,H) and we denote the img size as (H,W)
            normed_mobs.image = np.stack(images, axis=0).transpose(0, 3, 1, 2) # n, h, w, c -> n, c, h, w
            if original_dim == 5:
                normed_mobs.image = normed_mobs.image[:, np.newaxis, :, :, :]
        # convert MetaObs to policy-specific obs
        policy_obs = self.meta2obs(normed_mobs)
        # inference action
        action_chunk = self.policy.select_action(policy_obs)
        # (B, chunk_size, action_dim)
        macts = self.act2meta(action_chunk, ctrl_space=self.ctrl_space, ctrl_type=self.ctrl_type)
        action_chunk = macts.action
        is_chunked = (len(action_chunk.shape)==3)
        bs = action_chunk.shape[0]
        ac_dim = action_chunk.shape[-1]
        if is_chunked:
            macts.action = action_chunk.reshape(-1, ac_dim)
        # denormalize action
        macts = self.action_normalizer.denormalize_metaact(macts)
        if is_chunked:
            macts.action = macts.action.reshape(bs, -1, ac_dim).transpose(1, 0, 2)
        else:
            macts.action = macts.action[np.newaxis, :]
        mact_list = [np.array([asdict(MetaAction(action=aii, ctrl_type=macts.ctrl_type, ctrl_space=macts.ctrl_space)) for aii in ai], dtype=object) for ai in macts.action]
        # Only truncate if chunk_size is a positive number
        # chunk_size=-1 means use all actions
        if self.chunk_size is not None and self.chunk_size > 0:
            mact_list = mact_list[:self.chunk_size]
        return mact_list

    def select_action(self, mobs: MetaObs, t:int, return_all=False):
        # normalizing Obs and Actions
        # Determine if we need to infer new actions
        should_infer = len(self.action_queue) == 0  # Always infer when queue is empty
        
        # If chunk_size is valid (>0), also check if it's time to infer based on timestep
        if self.chunk_size is not None and self.chunk_size > 0:
            should_infer = should_infer or (t % self.chunk_size == 0)
        
        if should_infer:
            mobs.timestep = np.array([[t] for _ in range(mobs.state.shape[0])])
            mact_list = self.inference(mobs)
            while len(self.action_queue) > 0:
                self.action_queue.popleft()
            self.action_queue.extend(mact_list)
        # get action from queue
        if return_all:
            all_macts = []
            while len(self.action_queue) > 0:
                all_macts.append(self.action_queue.popleft())
            return np.concatenate(all_macts)
        mact = self.action_queue.popleft()
        return mact
    
    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        self.action_queue.clear()
    
    
    
