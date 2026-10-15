"""Small learned PointNav policy using the standard GPS/compass goal sensor."""
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_model(args):
    model = NavigationMLP.from_pretrained(args.model_name_or_path)
    return {'model': model.to(getattr(args, 'device', 'cpu'))}


def features(state):
    state = np.asarray(state, dtype=np.float32)
    distance, bearing = state[..., 0], state[..., 1]
    return np.stack([np.minimum(distance, 10.) / 10., np.sin(bearing), np.cos(bearing)], axis=-1).astype(np.float32)


class NavigationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64),
                                 nn.Tanh(), nn.Linear(64, 4))

    def forward(self, features):
        return self.net(features)

    def select_action(self, obs):
        state = obs['state']
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        with torch.inference_mode():
            logits = self(torch.as_tensor(features(state), device=next(self.parameters()).device))
        return logits.argmax(-1).float().reshape(-1, 1, 1)

    def predict(self, obs):
        state = obs.state if hasattr(obs, 'state') else obs['state']
        with torch.inference_mode():
            logits = self(torch.as_tensor(features(state), device=next(self.parameters()).device))
        return int(logits.argmax(-1).item())

    def inference(self, mobs):
        states = mobs.state
        if states.ndim == 1:
            states = states[None]
        with torch.inference_mode():
            actions = self(torch.as_tensor(features(states), device=next(self.parameters()).device)).argmax(-1).cpu().numpy()
        return [np.array([{'ctrl_space': 'navigation', 'ctrl_type': 'discrete',
                           'action': np.array([action], dtype=np.float32),
                           'gripper_continuous': False} for action in actions], dtype=object)]

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        target = Path(path)
        if target.is_dir():
            target = target / 'policy.pt'
        checkpoint = torch.load(target, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()
