from benchmark.base import MetaAction, MetaEnv, MetaObs
from libero.libero import benchmark as libero_bench
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
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

benchmark_dict = libero_bench.get_benchmark_dict()

class LiberoEnv(MetaEnv):
    def __init__(self, config, action_space='ee', *args):
        # 初始化env
        self.config = config
        self.action_space=action_space
        self.is_delta = True
        env = self.create_env()
        super().__init__(env)
        
    def create_env(self):
        task_id = self.config.task_id
        action_space = "OSC_POSE" if self.action_space=='ee' else "JOINT_POSITION"  # ee or joint
        task_suite = benchmark_dict[self.config.task]()
        init_states = task_suite.get_task_init_states(task_id)
        task = task_suite.get_task(task_id)
        self.raw_lang =  task.language
        self.task_name = task.name
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # step over the environment
        height, width = self.config.image_size
        self.image_size = self.config.image_size
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": height,
            "camera_widths": width,
            'controller_configs': load_controller_config(default_controller="JOINT_POSITION"),
        }
        env = OffScreenRenderEnv(**env_args)
        state = init_states[np.random.choice(len(init_states))]
        env.set_init_state(state)
        return env
        
    def process_act(self, maction: MetaAction):
        # MetaAct to action of libero
        # 先看MetaAct是什么类型的，再转成libero需要的类型: LIBERO 用的相对动作控制
        assert maction.atype==self.action_space, f"The atype of MetaAction {maction.atype} doesn't match the action space of environment {self.action_space}"
        # TODO: 如何MetaAct不是相对的，先转成相对的
        # 先假定action都是相对的
        assert maction.is_delta, "Action must be relative action"
        actions = maction.action # (chunk_size, action_dim=7)
        actions[:,:6] = actions[:,:6]/np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]) # LIBERO内部会乘上该缩放值实现控制，所以提前抵消该缩放值保证能够正确执行
        actions[:,6] = 1.-2.*actions[:,6] # LIBERO使用01二元控制信号，实现
        return actions
        
    def process_obs(self, obs):
        # ee state
        xyz = obs['robot0_eef_pos'] # (3,)
        euler = quat2axisangle(obs['robot0_eef_quat']) # (3,)
        ee_state = np.concatenate([xyz, euler], axis=0)
        # joint state
        joint_state = obs["robot0_joint_pos"]
        # gripper state
        gpos = obs['robot0_gripper_qpos']
        gripper_state = np.array([(gpos[0]-gpos[1])/0.08]) # (1,)
        # image
        img_primary = obs["agentview_image"][::-1, ::-1]
        img_second = obs['robot0_eye_in_hand_image']
        image = np.stack([img_primary, img_second])
        # depth
        depth_primary = obs["agentview_depth"][::-1, ::-1]
        depth_second = obs['robot0_eye_in_hand_depth']
        depth = np.stack([depth_primary, depth_second])
        return MetaObs(ee_state=ee_state, joint_state=joint_state, gripper_state=gripper_state, image=image, depth=depth, raw_lang=self.raw_lang)
    
    
