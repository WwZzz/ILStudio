from benchmark.base import MetaEnv
from benchmark.act_mujoco.constants import SIM_TASK_CONFIGS
from benchmark.act_mujoco.ee_sim_env import make_ee_sim_env
from benchmark.act_mujoco.sim_env import make_sim_env, BOX_POSE
from benchmark.act_mujoco.utils import sample_box_pose, sample_insertion_pose
import numpy as np

class ACTEnv(MetaEnv):
    def __init__(self, task_name='sim_transfer_cube_scripted'):
        super().__init__()
        self.task_name = task_name
        self.max_steps = SIM_TASK_CONFIGS[self.task_name]['episode_len']
        self.camera_names = SIM_TASK_CONFIGS[self.task_name]['camera_names']
        self.env = make_sim_env(self.task_name)

    def reset(self, *args, **kwargs):
        if 'sim_transfer_cube' in self.task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'sim_insertion' in self.task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        ts = self.env.reset()
        return ts.observation

    def step(self, action):
        ts = self.env.step(action)
        return {'obs':ts.observation, 'reward':ts.reward, 'done':self.env.task.max_reward==ts.reward, 'info':ts}

    def get_task_prompt(self):
        return self.task_name

