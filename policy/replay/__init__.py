from data_utils.utils import EpisodicDataset, load_normalizer_from_meta
from configs.task.loader import load_task_config
from dataclasses import dataclass
import importlib
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
    ctrl_type: str = 'delta'

class Replay(nn.Module):
    def __init__(self, actions, ctrl_type='delta', ctrl_space='ee'):
        super().__init__()
        self.actions = torch.stack(actions).numpy()
        self.ctrl_type = ctrl_type
        self.ctrl_space = ctrl_space
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
    task_config = load_task_config(args.task)
    args.camera_names = task_config['camera_names']
    with open(args.norm_path, 'r') as f:
        norm_meta = json.load(f)
    normalizers = load_normalizer_from_meta(task_config['dataset_dir'][0], norm_meta)
    data_class = task_config.get('dataset_class', 'EpisodicDataset')
    if data_class == 'EpisodicDataset':
        from data_utils.datasets import EpisodicDataset
        data_class = EpisodicDataset
    else:
        data_class = getattr(importlib.import_module('data_utils.datasets'), data_class)
    data = data_class(
        [args.model_name_or_path], 
        camera_names=args.camera_names, 
        action_normalizers={task_config['dataset_dir'][0]:normalizers['action']}, 
        state_normalizers={task_config['dataset_dir'][0]:normalizers['state']}, 
        data_args=DataArgs(chunk_size=args.chunk_size, ctrl_type=args.ctrl_type), 
        chunk_size = args.chunk_size,
        ctrl_space=args.ctrl_space,
        ctrl_type=args.ctrl_type,
    )
    all_actions = [data[i]['action'] for i in range(len(data))]
    return {'model': Replay(all_actions, ctrl_type=args.ctrl_type, ctrl_space=args.ctrl_space)}
