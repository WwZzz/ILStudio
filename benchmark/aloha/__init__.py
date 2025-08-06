"""
Observation Format
===============================================================================================================
sim_transfer_cube_scripted\sim_insertion_scripted
---------------------------------------------------------------------------------------------------------------
obs [dict] = {
    'qpos': ndarray:(14,),
    'qvel': ndarray:(14,),
    'env_state': ndarray:(7,),
    'images': {
        'top': ndarray:(480, 460, 3),
        'angle': ndarray:(480, 460, 3),
        'vis': ndarray:(480, 460, 3),
    }
}
---------------------------------------------------------------------------------------------------------------
act [ndarray(14,)]
---------------------------------------------------------------------------------------------------------------
===============================================================================================================
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""
from numpy.core.tests.test_mem_overlap import shape
from benchmark.base import MetaEnv, MetaAction, MetaObs, MetaPolicy
from .constants import SIM_TASK_CONFIGS
from .ee_sim_env import make_ee_sim_env
from .sim_env import make_sim_env, BOX_POSE
from .utils import sample_box_pose, sample_insertion_pose
import numpy as np
from dataclasses import dataclass, field, fields, asdict
from data_utils.rotate import quat2axisangle
import os
import numpy as np
from torchvision import transforms
import pickle
import time
import copy
import json
import tensorflow as tf
from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio
from robosuite.controllers import load_controller_config


TASK_PROMPT = {
    'sim_transfer_cube_scripted': 'Transfer the red cube to the other arm.',
    'sim_insertion_scripted': 'Insert the red peg into the blue socket.'
}

def create_env(config):
    config.task
    return ACTEnv(config, action_space=config.space_name)

class ACTEnv(MetaEnv):
    def __init__(self, config, action_space='ee', *args):
        # 初始化env
        self.config = config
        self.action_space=action_space
        self.abs_control = False
        env = self.create_env()
        super().__init__(env)
        
    def create_env(self):
        task_name = self.config.task
        
        task_name = task_info[0] + '_' + task_info[1] # libero_{object, goal, spatial, 10, 90}
        task_id = int(task_info[-1])
        action_space = "OSC_POSE" if self.action_space=='ee' else "JOINT_POSITION"  # ee or joint
        task_suite = benchmark_dict[task_name]()
        init_states = task_suite.get_task_init_states(task_id)
        task = task_suite.get_task(task_id)
        self.raw_lang =  task.language
        self.task_name = task.name
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # step over the environment
        image_size = eval(self.config.image_size)
        if isinstance(image_size, tuple):
            height, width = image_size
        elif isinstance(image_size, int):
            height, width = image_size, image_size
        else:
            raise ValueError("image_size should be str either '(height, width)' or 'height'")
        self.image_size = (height, width)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": height,
            "camera_widths": width,
        }
        env = OffScreenRenderEnv(**env_args)
        state = init_states[np.random.choice(len(init_states))]
        env.set_init_state(state)
        return env
        
    def meta2act(self, maction: MetaAction):
        # MetaAct to action of libero
        # 先看MetaAct是什么类型的，再转成libero需要的类型: LIBERO 用的相对动作控制
        assert maction['space_name']==self.action_space, f"The space_name of MetaAction {maction['space_name']} doesn't match the action space of environment {self.action_space}"
        # TODO: 如何MetaAct不是相对的，先转成相对的
        # 先假定action都是相对的
        assert maction['is_delta'], "Action must be relative action for LIBERO"
        actions = maction['action'] # (action_dim, )
        actions[:6] = actions[:6]/np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]) # LIBERO内部会乘上该缩放值实现控制，所以提前抵消该缩放值保证能够正确执行
        actions[6] = 1.-2.*actions[6] # LIBERO使用01二元控制信号，实现
        return actions
        
    def obs2meta(self, obs):
        # gripper state
        gpos = obs['robot0_gripper_qpos']
        gripper_state = np.array([(gpos[0]-gpos[1])/0.08]) # (1,)
        # ee state
        xyz = obs['robot0_eef_pos'] # (3,)
        euler = quat2axisangle(obs['robot0_eef_quat']) # (3,)
        state_ee = np.concatenate([xyz, euler, gripper_state], axis=0).astype(np.float32)
        # joint state
        state_joint = np.concatenate([obs["robot0_joint_pos"], gripper_state], axis=0).astype(np.float32)
        # image
        img_primary = obs["agentview_image"][::-1, ::-1]
        img_second = obs['robot0_eye_in_hand_image']
        image = np.stack([img_primary, img_second])
        image = image.transpose(0, 3, 1, 2)
        return MetaObs(state_ee=state_ee, state_joint=state_joint, image=image, raw_lang=self.raw_lang)
    
    



class ACTEnv(MetaEnv):
    ALL_OBSERVATION_KEYS = ['qpos', 'qvel', 'env_state', 'images']
    def __init__(self, task_name='sim_transfer_cube_scripted', obs_map={'qpos': 'observation.state', 'images.top': 'observation.images.top'}):
        super().__init__()
        self.task_name = task_name
        self.max_steps = SIM_TASK_CONFIGS[self.task_name]['episode_len']
        self.camera_names = SIM_TASK_CONFIGS[self.task_name]['camera_names']
        self.env = make_sim_env(self.task_name)
        self.obs_map = obs_map

    def process_observation(self, obs):
        return {'observation.state': obs['qpos'], 'observation.images': obs['images']['top']}

    def reset(self, *args, **kwargs):
        if 'sim_transfer_cube' in self.task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'sim_insertion' in self.task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        ts = self.env.reset()
        return self.process_observation(ts.observation)

    def step(self, action):
        ts = self.env.step(action)
        return {'obs':self.process_observation(ts.observation), 'reward':ts.reward, 'done': self.env.task.max_reward == ts.reward, 'info':ts}

    def get_task_prompt(self):
        return TASK_PROMPT[self.task_name]

    def get_info(self):
        return {
            'action': PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
            'observation.state':  PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            'observation.images': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        }
