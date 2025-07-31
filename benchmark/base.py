import numpy as np
from dataclasses import dataclass, field, fields, asdict

@dataclass
class MetaAction:
    atype: str = 'ee' # or 'joint'
    is_delta: bool = True # absolute position control or relative
    action: np.ndarray = None # action[-1] is gripper control signal, i.e., 1 is open and 0 is close
    gripper_continuous: bool = False # is gripper controlled by continuous action, where action[-1] is position ratio to the gripper width

@dataclass
class MetaObs:
    ee_state : np.ndarray = None
    joint_state: np.ndarray = None
    gripper_state: np.ndarray = None
    obj_state: np.ndarray = None
    image: np.ndarray = None # (K, C, H, W)
    depth: np.ndarray = None # (K, H, W)
    pc: np.array = None # (n, 3)
    raw_lang: str = ''

class MetaEnv:
    def __init__(self, env):
        self.env = env
        self.prev_obs = None
        
    def step(self, *args, **kwargs):
        processed_action = self.process_action(*args, **kwargs)
        results = list(self.env.step(**processed_action))
        self.prev_obs = results[0] = self.process_obs(results[0])
        return tuple(results)
    
    def process_obs(self, raw_obs):
        # convert raw_obs into MetaObs
        raise NotImplementedError
    
    def process_act(self, action, *args):
        # convert MetaAction into env-specific action
        raise NotImplementedError
    
    def reset(self):
        res = self.env.reset()
        self.prev_obs = self.process_obs(init_obs)
        return res
    
    def close(self):
        self.env.close()
    
class MetaPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.action_queue = None
    
    def process_obs(self, obs: MetaObs):
        # convert MetaObs into policy-specific obs
        return NotImplementedError
    
    def process_act(self, action):
        # convert action 
        return NotImplementedError
    
