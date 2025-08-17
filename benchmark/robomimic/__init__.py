import robomimic.utils.env_utils as EnvUtils
import sys
from tqdm import tqdm
import torch
# from eval.utils import run_rollout, setup_seed
# from eval.config import ALL_ENV_CONFIGS
import argparse
import numpy as np
from collections import defaultdict
import json
import robomimic.utils.obs_utils as ObsUtils
from benchmark.robomimic.constant import ALL_ENV_CONFIGS, ALL_ENV_LANGUAGES
from data_utils.rotate import quat2axisangle
from ..base import *
from .constant import ALL_ENV_CONFIGS

ALL_TASKS = ['Lift_Panda', "PickPlaceCan_Panda", "NutAssemblySquare_Panda", "ToolHang_Panda", "TwoArmTransport_Panda"]

def create_env(config):
    return RobomimicEnv(config, ctrl_space='ee')

class RobomimicEnv(MetaEnv):
    def __init__(self, config, ctrl_space='ee', *args):
        # 初始化env
        self.config = config
        self.ctrl_space = ctrl_space
        self.ctrl_type = 'delta'
        env = self.create_env()
        super().__init__(env)
        
    def create_env(self):
        task_info = self.config.task.split('_')
        env_name, robot_name = task_info[0], task_info[1]
        env_meta = ALL_ENV_CONFIGS[env_name][robot_name]
        self.raw_lang = ALL_ENV_LANGUAGES.get(env_name, '')
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            use_depth_obs=False,
        )
        modalities = {
            'obs':{
                "low_dim": [x for x in env.base_env.observation_names if 'image' not in x and 'depth' not in x],
                "rgb": [x for x in env.base_env.observation_names if 'image' in x],
                "depth": [x for x in env.base_env.observation_names if 'depth' in x],
                "scan": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(modalities)
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        assert maction['ctrl_space']==self.ctrl_space, f"The ctrl_space of MetaAction {maction['ctrl_space']} doesn't match the action space of environment {self.ctrl_space}"
        assert maction['ctrl_type']==self.ctrl_type, "Action must be delta action for LIBERO"
        actions = maction['action'] # (action_dim, )
        actions[:6] = actions[:6]/np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]) # Robosuite内部会乘上该缩放值实现控制，所以提前抵消该缩放值保证能够正确执行
        actions[6] = 1.-2.*actions[6]
        return actions
        
    def obs2meta(self, obs):
        # gripper state
        gpos = obs['robot0_gripper_qpos']
        gripper_state = np.array([(gpos[0]-gpos[1])]) # (1,)
        # ee state
        xyz = obs['robot0_eef_pos'] # (3,)
        euler = quat2axisangle(obs['robot0_eef_quat']) # (3,)
        state_ee = np.concatenate([xyz, euler, gripper_state], axis=0).astype(np.float32)
        # joint state
        state_joint = np.concatenate([obs["robot0_joint_pos"], gripper_state], axis=0).astype(np.float32)
        # image
        img_primary = obs["agentview_image"]
        img_second = obs['robot0_eye_in_hand_image']
        image = np.stack([img_primary, img_second])
        image = (image*255.0).astype(np.uint8)
        return MetaObs(state=state_ee, image=image, raw_lang=self.raw_lang)

