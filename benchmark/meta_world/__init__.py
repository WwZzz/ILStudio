import time
import numpy as np
from ..base import *
import gymnasium as gym
from multiprocessing import current_process
import metaworld
from .task_desc import TASK_DESC

ALL_CAMERA_NAMES = ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
ALL_TASKS = metaworld.ALL_V3_ENVIRONMENTS

def create_env(config):
    return MetaWorldEnv(config)

class MetaWorldEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.ctrl_type = 'abs'
        self.render_mode = getattr(self.config, 'render_mode', 'rgb_array')
        self.camera_name = getattr(self.config, 'camera_name', None)
        assert self.camera_name is None or self.camera_name in ALL_CAMERA_NAMES
        self.raw_lang = TASK_DESC[self.config.task][1]
        env = self.create_env()
        super().__init__(env)
    
    def create_env(self):
        task = self.config.task
        if self.camera_name is not None:
            env = gym.make(task, render_mode=self.render_mode, camera_name=self.camera_name)
        else:
            env = gym.make(task, render_mode=self.render_mode)
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        # assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        # assert maction['ctrl_type']==self.ctrl_type, "Action must be abs action for PandaGym"
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state = obs.astype(np.float32)    
        if self.use_camera:
            # PandaGym only has one camera view from env.render()
            image = self.env.render()
            if image is not None:
                # Convert to (N, C, H, W) format for consistency with other environments
                if len(image.shape) == 3:  # (H, W, C)
                    image = image[np.newaxis, ...]  # Add batch dimension -> (1, H, W, C)
                if image.shape[-1] == 3:  # (N, H, W, C) -> (N, C, H, W)
                    image = image.transpose(0, 3, 1, 2)
        else:
            image = None
        return MetaObs(state=state, image=image, raw_lang=self.raw_lang)

    def step(self, *args, **kwargs):
        action = args[0]['action']
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs = self.obs2meta(observation)
        done = info['success']
        info['terminated'] = terminated
        info['truncated'] = truncated
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return self.obs2meta(obs[0])