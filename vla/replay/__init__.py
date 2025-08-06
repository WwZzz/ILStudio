from data_utils.utils import EpisodicDataset, load_normalizer_from_meta
from configuration.constants import TASK_CONFIGS
from dataclasses import dataclass
import torch.nn as nn
import json
import torch
import numpy as np

@dataclass
class DataArgs:
    use_prev_subtask: bool = False
    chunk_size: int = 1
    use_reasoning: bool = False
    image_size_wrist: str = "(256, 256)"
    image_size_primary: str= "(256, 256)"
    abs_control: bool = False

class Replay(nn.Module):
    def __init__(self, actions, abs_control=False, space_name='ee'):
        super().__init__()
        self.actions = torch.stack(actions).numpy()
        self.abs_control = abs_control
        self.space_name = space_name
        self.count = 0
        
    def reset(self):
        self.count = 0
                
    def select_action(self, obs):
        bs = obs['state'].shape[0]
        act = self.actions[self.count%self.actions.shape[0]]
        batch_act = np.tile(act, (bs, 1))
        self.count += 1
        return batch_act

def load_model(args):
    task_config = TASK_CONFIGS[args.task]
    args.camera_names = task_config['camera_names']
    with open(args.norm_path, 'r') as f:
        norm_meta = json.load(f)
    normalizers = load_normalizer_from_meta(task_config['dataset_dir'][0], norm_meta)
    data = EpisodicDataset(
        [args.model_name_or_path], 
        camera_names=args.camera_names, 
        action_normalizers={task_config['dataset_dir'][0]:normalizers['action']}, 
        state_normalizers={task_config['dataset_dir'][0]:normalizers['state']}, 
        data_args=DataArgs(chunk_size=args.chunk_size, abs_control=args.abs_control), 
        control_space=args.space_name
        )
    all_actions = [data[i]['action'] for i in range(len(data))]
    return {'model': Replay(all_actions, abs_control=args.abs_control, space_name=args.space_name)}
