import numpy as np
import cv2
from dataclasses import dataclass, field, fields, asdict
from collections import deque
from typing import Optional
import torch
from loguru import logger
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
    def __init__(self, policy, chunk_size:Optional[int] = None, action_normalizer=None, state_normalizer=None, ctrl_space='ee', ctrl_type='delta'):
        """
        MetaPolicy wrapper for policy inference.
        
        Note: Image resizing is now handled by each policy's data_processor,
              not in MetaPolicy. This ensures consistent data processing between
              training and inference.
        
        Args:
            policy: The policy model
            chunk_size: Action chunk size for chunked action prediction
            action_normalizer: Normalizer for actions
            state_normalizer: Normalizer for states  
            ctrl_space: Control space ('ee' or 'joint')
            ctrl_type: Control type ('delta' or 'abs')
        """
        self.policy = policy
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.action_queue = deque(maxlen=chunk_size) if (chunk_size is not None and chunk_size>0) else deque(maxlen=1000)
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer

    def meta2obs(self, samples: list):
        """
        Convert standard format samples to policy-specific observation format.
        
        This method applies policy's data_processor and data_collator to transform
        samples into the format used during training, ensuring consistency.
        
        Args:
            samples: List of sample dicts in ILStudio standard format, each containing:
                - 'image': (cameras, C, H, W) or (C, H, W) - raw image from environment
                - 'state': (state_dim,) - state vector
                - 'raw_lang': str - language instruction
                - 'timestamp': int (optional) - timestep
        
        Returns:
            Policy-specific observation format (processed and collated batch)
        """
        # Check if policy has custom meta2obs (legacy support)
        if hasattr(self.policy, 'meta2obs') and callable(self.policy.meta2obs):
            return self.policy.meta2obs(samples)
        
        # Standard processing pipeline: data_processor → data_collator
        processed_samples = samples
        
        # Step 1: Apply data_processor if available
        if hasattr(self.policy, 'data_processor') and self.policy.data_processor is not None:
            processed_samples = [self.policy.data_processor(sample) for sample in samples]
        
        # Step 2: Apply data_collator if available
        if hasattr(self.policy, 'data_collator') and self.policy.data_collator is not None:
            batch_obs = self.policy.data_collator(processed_samples)
        else:
            # Default collator: simple batching with torch
            batch_obs = self._default_collate(processed_samples)
        
        return batch_obs
    
    def _default_collate(self, samples: list):
        """
        Default collator when policy doesn't provide one.
        Simply converts numpy arrays to tensors and stacks them.
        """
        if not samples:
            return {}
        
        batch = {}
        keys = samples[0].keys()
        
        for key in keys:
            values = [s[key] for s in samples]
            
            # Skip None values
            if values[0] is None:
                batch[key] = None
                continue
            
            # Handle numpy arrays and tensors
            if isinstance(values[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            # Handle strings (like raw_lang)
            elif isinstance(values[0], str):
                batch[key] = values  # Keep as list
            # Handle scalars
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        
        return batch
    
    def act2meta(self, action, ctrl_space:str='ee', ctrl_type:str=True):
        # convert action into MetaAction, np.ndarray((chunk_size, action_dim), dtype=np.float32) as default
        if hasattr(self.policy, 'act2meta'):
            mact = self.policy.act2meta(action, ctrl_space=ctrl_space, ctrl_type=ctrl_type)
        else:
            if isinstance(action, torch.Tensor): action = action.detach().float().cpu().numpy()
            mact = MetaAction(action=action, ctrl_space=ctrl_space, ctrl_type=ctrl_type) # (B, chunk_size, dim) or (chunk_size, dim)
        return mact 
    
    def is_action_queue_empty(self):
        return len(self.action_queue)==0

    def normed_mobs_to_samples(self, normed_mobs: MetaObs):
        """
        Convert normalized MetaObs to samples in ILStudio standard format.
        """
         # Prepare standard format samples
        batch_size = normed_mobs.state.shape[0] if normed_mobs.state is not None and len(normed_mobs.state.shape) > 1 else 1
        
        samples = []
        for i in range(batch_size):
            sample = {}
            
            # Image: keep original format, let policy's data_processor handle resize
            if normed_mobs.image is not None:
                if normed_mobs.image.ndim == 5:  # (batch, cameras, C, H, W)
                    sample['image'] = normed_mobs.image[i]  # (cameras, C, H, W)
                elif normed_mobs.image.ndim == 4:  # (batch, C, H, W) or (cameras, C, H, W)
                    if batch_size > 1:
                        sample['image'] = normed_mobs.image[i]  # (C, H, W)
                    else:
                        sample['image'] = normed_mobs.image  # (cameras, C, H, W)
                else:
                    sample['image'] = normed_mobs.image
            else:
                sample['image'] = None  # Explicitly set to None when no image
            
            # State: use specified control space (already normalized)
            sample['state'] = normed_mobs.state[i] if len(normed_mobs.state.shape) > 1 else normed_mobs.state
            
            # Language instruction
            if normed_mobs.raw_lang is not None:
                if isinstance(normed_mobs.raw_lang, list):
                    sample['raw_lang'] = normed_mobs.raw_lang[i] if len(normed_mobs.raw_lang) > i else normed_mobs.raw_lang[0]
                else:
                    sample['raw_lang'] = normed_mobs.raw_lang
            
            # Timestamp (optional)
            if hasattr(normed_mobs, 'timestep') and normed_mobs.timestep is not None:
                sample['timestamp'] = normed_mobs.timestep[i] if len(normed_mobs.timestep.shape) > 1 else normed_mobs.timestep
            # Construct observations, convert arrays to tensors
            if sample['image'] is not None:
                # Safety check: ensure images are uint8 with values in [0, 255]
                from data_utils.utils import ensure_uint8_image
                sample['image'] = ensure_uint8_image(sample['image'])   # (cameras, C, H, W)
                sample['image'] = torch.from_numpy(sample['image'])  # (cameras, C, H, W)
            else:
                sample['image'] = None
            sample['state'] = torch.from_numpy(sample['state']).float() if 'state' in sample else None  # (state_dim,)
            if 'action' in sample:
                sample['action'] = torch.from_numpy(sample['action']).float() 
            if 'is_pad' in sample:
                sample['is_pad'] = torch.from_numpy(sample['is_pad']).bool() 
            samples.append(sample)
        return samples

    def inference(self, mobs: MetaObs):
        """
        Prepare observation data in ILStudio standard format for policy inference.
        
        The standard format is a list of samples, where each sample is a dict containing:
        - 'image': image data (N, C, H, W) or (N, num_cameras, C, H, W)
        - 'state': state data (from ctrl_space: 'ee' or 'joint') - normalized
        - 'raw_lang': language instruction
        - 'timestamp': optional timestep information
        
        This format matches training data format, allowing policy to use the same
        data_processor and collator for consistent data handling.
        """
        # Normalize state before preparing samples
        normed_mobs = self.state_normalizer.normalize_metaobs(mobs, self.ctrl_space)
        
        samples = self.normed_mobs_to_samples(normed_mobs)
        
        # Convert samples to policy-specific format using policy's data processing
        policy_obs = self.meta2obs(samples)
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
        # print(macts.action)
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
    
    
    
