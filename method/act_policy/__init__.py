import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from method import MetaAgent
from method.act_policy.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from benchmark.act_mujoco.constants import SIM_TASK_CONFIGS
import IPython
import numpy as np
import os
import pickle
from einops import rearrange
import torch
e = IPython.embed

def get_image(obs, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

"""
--kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
"""

class ACTAgent(MetaAgent):
    def __init__(self, ckpt_path, task_name='sim_transfer_cube_scripted', temporal_agg=False, state_dim=14, chunk_size=100, max_steps=400, device='cuda'):
        super().__init__()
        self.temporal_agg = temporal_agg
        self.max_timesteps = max_steps
        self.state_dim = state_dim
        self.task_config = SIM_TASK_CONFIGS[task_name]
        self.ckpt_path = ckpt_path
        self.ckpt_dir = os.path.dirname(ckpt_path)
        self.camera_names = self.task_config['camera_names']
        self.query_frequency = chunk_size
        self.device = device
        with open(os.path.join(self.ckpt_dir, f'dataset_stats.pkl'), 'rb') as f:
            self.stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        self.policy_config = {
            'lr': 1e-5,
            'kl_weight': 10,
            'chunk_size': chunk_size,
            'num_queries': chunk_size,
            'hidden_dim': 512,
            'batch_size': 8,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': self.camera_names,
        }

        self.policy = ACTModel(self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        self.policy.to(device)
        if temporal_agg:
            self.query_frequency = 1
            self.num_queries = chunk_size
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).to(self.device)
        self.all_actions = None

    def reset(self):
        self.all_actions = None
        if self.temporal_agg: self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).to(self.device)

    def process_observation(self, obs, *args, **kwargs):
        qpos_numpy = np.array(obs['qpos'])
        qpos = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        curr_image = get_image(obs, self.camera_names)
        return {'image': curr_image, 'qpos': qpos}

    def act(self, data):
        obs, t = data['obs'], data['t']
        if t % self.query_frequency == 0:
            self.all_actions = self.policy(obs['qpos'], obs['image'])
        if self.temporal_agg:
            self.all_time_actions[[t], t:t + self.num_queries] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            raw_action = self.all_actions[:, t % self.query_frequency]
        raw_action = raw_action.squeeze(0).detach().cpu().numpy()
        action = self.post_process(raw_action)
        return action

class ACTModel(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
