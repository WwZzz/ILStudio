"""Official VLN-CE CMA policy using RGB, normalized depth, and instruction tokens.

The recurrent state is owned by the policy. Goal sensors, reference paths and
expert actions are never policy inputs. See vendor/PROVENANCE.json.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from gym import spaces
from yacs.config import CfgNode

from .vendor.cma import CMANet


class CategoricalHead(nn.Module):
    def __init__(self, hidden_size, action_count):
        super().__init__()
        self.linear = nn.Linear(hidden_size, action_count)


class CMAPolicy(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        observation_space = spaces.Dict({
            'rgb': spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
            'depth': spaces.Box(0., 1., shape=(256, 256, 1), dtype=np.float32),
            'instruction': spaces.Box(0, 2503, shape=(200,), dtype=np.int64)})
        self.net = CMANet(observation_space, CfgNode(model_config), num_actions=4)
        self.net.instruction_encoder.embedding_layer.weight.requires_grad_(
            bool(model_config['INSTRUCTION_ENCODER'].get('fine_tune_embeddings', False)))
        self.action_distribution = CategoricalHead(self.net.output_size, 4)
        self.reset()

    def reset(self):
        self.hidden = None
        self.prev_actions = None

    def initial_state(self, batch_size):
        return torch.zeros(batch_size, self.net.num_recurrent_layers,
                           self.net.output_size, device=next(self.parameters()).device)

    def forward(self, observations, hidden, previous_actions, masks):
        features, hidden = self.net(observations, hidden, previous_actions, masks)
        return self.action_distribution.linear(features), hidden

    def select_action(self, obs):
        device = next(self.parameters()).device
        rgb = torch.as_tensor(obs['image'], device=device)
        depth = torch.as_tensor(obs['depth'], device=device, dtype=torch.float32)
        if rgb.ndim != 5 or tuple(rgb.shape[1:]) != (1, 3, 224, 224):
            raise ValueError('CMA requires one 224x224 RGB camera')
        if depth.ndim != 4 or tuple(depth.shape[1:]) != (1, 256, 256):
            raise ValueError('CMA requires one normalized 256x256 depth camera')
        if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
            raise ValueError('CMA depth must be finite and normalized to [0,1] using 0-10m')
        rows = obs.get('navigation')
        if not rows or any('instruction_tokens' not in row for row in rows):
            raise ValueError('CMA requires return_instruction_tokens=true and the original R2R vocabulary')
        tokens = torch.as_tensor([row['instruction_tokens'] for row in rows], device=device, dtype=torch.long)
        masks = (torch.as_tensor(obs['timestamp'], device=device).reshape(-1, 1) > 0)
        if self.hidden is None or self.hidden.shape[0] != rgb.shape[0]:
            self.hidden = self.initial_state(rgb.shape[0])
            self.prev_actions = torch.zeros(rgb.shape[0], 1, device=device, dtype=torch.long)
        observations = {'rgb': rgb[:, 0].permute(0, 2, 3, 1),
                        'depth': depth[:, 0, :, :, None], 'instruction': tokens}
        with torch.inference_mode():
            logits, self.hidden = self(observations, self.hidden, self.prev_actions, masks)
            self.prev_actions = logits.argmax(-1, keepdim=True)
        return self.prev_actions.float().reshape(-1, 1, 1)

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        path = Path(path)
        checkpoint = torch.load(path / 'policy.pt' if path.is_dir() else path,
                                map_location='cpu', weights_only=True)
        model = cls(checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.to(device).eval()


def load_model(args):
    return {'model': CMAPolicy.from_pretrained(args.model_name_or_path, getattr(args, 'device', 'cpu'))}
