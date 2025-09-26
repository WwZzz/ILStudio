import time
import numpy as np
from ..base import *
import gymnasium as gym
from multiprocessing import current_process
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

def create_env(config):
    return PandaGymEnv(config)

class PandaGymEnv(MetaEnv):
    def __init__(self, config, *args):
        # 初始化env，仅从 config 读取参数
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.ctrl_type = 'abs'
        self.use_camera = getattr(self.config, 'use_camera', True)
        self.render_mode = getattr(self.config, 'render_mode', 'rgb_array')
        self.renderer = getattr(self.config, 'renderer', 'Tiny') # or OpenGL
        self.use_dense_reward = getattr(self.config, 'use_dense_reward', False)
        self.raw_lang = self.config.task[5:] # remove 'Panda' from task name
        env = self.create_env()
        super().__init__(env)
    
    def create_env(self):
        task = self.config.task
        env = gym.make(task, render_mode=self.render_mode, renderer=self.renderer)
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        assert maction['ctrl_type']==self.ctrl_type, "Action must be abs action for PandaGym"
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state = obs['observation']        
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
        observation, reward, terminated, truncated, info = super().step(*args, **kwargs)
        done = bool(info['is_success'])
        info['terminated'] = terminated
        info['truncated'] = truncated
        return observation, reward, done, info
    
    def reset(self):
        pid = current_process().pid  # 获取当前进程 ID
        seed = (pid * 1000 + time.time_ns()) % (2**32)  # 基于时间戳生成种子
        np.random.seed(seed)
        return super().reset()