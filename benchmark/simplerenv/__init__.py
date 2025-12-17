import os
os.environ['MS2_REAL2SIM_ASSET_DIR'] = os.path.join(os.path.dirname(__file__), 'ManiSkill2_real2sim', 'data')
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import numpy as np
from ..base import *
from multiprocessing import current_process

def create_env(config):
    return SimplerEnv(config)

class SimplerEnv(MetaEnv):
    def __init__(self, config, *args):
        self.config = config
        self.ctrl_space = getattr(self.config, 'ctrl_space', 'joint')
        self.ctrl_type = 'abs'
        self.max_timesteps = getattr(self.config, 'max_timesteps', 200)
        env = self.create_env()
        if "google_robot" in env.robot_uid:
            self.camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            self.camera_name = "3rd_view_camera"
        self.raw_lang = env.get_language_instruction()
        super().__init__(env)

    
    def create_env(self):
        task = self.config.task
        env = simpler_env.make(task)
        return env
        
    def meta2act(self, maction: MetaAction):
        actions = maction['action'] # (action_dim, )
        return actions
        
    def obs2meta(self, obs):
        state = obs['agent']['qpos']
        image = np.stack([obs['image'][self.camera_name]['rgb']])
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
        obs, reset_info = self.env.reset()
        return self.obs2meta(obs)


