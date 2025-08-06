import numpy as np
from dataclasses import dataclass, field, fields, asdict
from collections import deque
import torch

@dataclass
class MetaAction:
    space_name: str = 'ee' # or 'joint'
    is_delta: bool = True # absolute position control or relative
    action: np.ndarray = None # action[-1] is gripper control signal, i.e., 1 is open and 0 is close
    gripper_continuous: bool = False # is gripper controlled by continuous action, where action[-1] is position ratio to the gripper width

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __len__(self):
        if self.action is None: return 0
        if len(self.action.shape)==1: return 1
        return self.action.shape[0]
    
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
    depth: np.ndarray = None
    
    def __getitem__(self, key):
        return getattr(self, key)
    
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
        # convert raw_obs into MetaObs
        raise NotImplementedError
    
    def meta2act(self, action, *args):
        # convert MetaAction into env-specific action
        
        raise NotImplementedError
    
    def reset(self):
        init_obs = self.env.reset()
        self.prev_obs = self.obs2meta(init_obs)
        return self.prev_obs
    
    def close(self):
        self.env.close()
    
class MetaPolicy:
    def __init__(self, policy, freq:int, action_normalizer=None, state_normalizer=None, ctrl_space='ee', abs_ctrl=False):
        self.policy = policy
        self.freq = freq
        self.ctrl_space = ctrl_space
        self.abs_ctrl = abs_ctrl
        self.action_queue = deque(maxlen=freq)
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
    
    def meta2obs(self, mobs: MetaObs):
        # convert MetaObs into policy-specific obs
        if hasattr(self.policy, 'meta2obs'):
            obs = self.policy.meta2obs(mobs)
        else:
            mobs.state = mobs['state_ee'] if self.ctrl_space=='ee' else self.ctrl_space=='joint'
            obs = asdict(mobs)
        return obs
    
    def act2meta(self, action, space_name:str='ee', is_delta:bool=True):
        # convert action into MetaAction, np.ndarray((chunk_size, action_dim), dtype=np.float32) as default
        if hasattr(self.policy, 'act2meta'):
            mact = self.policy.act2meta(action, space_name=space_name, is_delta=is_delta)
        else:
            if isinstance(action, torch.Tensor): action = action.float().cpu().numpy()
            mact = MetaAction(action=action, space_name=space_name, is_delta=is_delta) # (B, chunk_size, dim) or (chunk_size, dim)
        return mact 
    
    def select_action(self, mobs: MetaObs, t:int):
        if t % self.freq == 0 or len(self.action_queue)==0:
            # 归一化观测
            normed_mobs = self.state_normalizer.normalize_metaobs(mobs, self.ctrl_space)
            # 转换MetaObs
            policy_obs = self.meta2obs(normed_mobs)
            # 推理动作
            action_chunk = self.policy.select_action(policy_obs)
            # 动作转为MetaAction  (B, chunk_size, action_dim)
            macts = self.act2meta(action_chunk, space_name=self.ctrl_space, is_delta=not self.abs_ctrl)
            action_chunk = macts.action
            is_chunked = (len(action_chunk.shape)==3)
            bs = action_chunk.shape[0]
            ac_dim = action_chunk.shape[-1]
            if is_chunked:
                macts.action = action_chunk.reshape(-1, ac_dim)
            # 反归一化动作
            macts = self.action_normalizer.denormalize_metaact(macts)
            if is_chunked:
                macts.action = macts.action.reshape(bs, -1, ac_dim).transpose(1, 0, 2)
            else:
                macts.action = macts.action[np.newaxis, :]
            while len(self.action_queue) > 0:
                self.action_queue.popleft()
            mact_list = [np.array([asdict(MetaAction(action=aii, is_delta=macts.is_delta, space_name=macts.space_name)) for aii in ai], dtype=object) for ai in macts.action]
            self.action_queue.extend(mact_list)
        # 从队列里拿动作
        mact = self.action_queue.popleft()
        return mact
    
    
    
    
    
