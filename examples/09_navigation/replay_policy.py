"""Diagnostic action replay for testing transport; never a learned VLN baseline.

Plans must come from verify_vln.py on the exact same assets and configuration.
The policy intentionally fails on unknown instructions or an exhausted plan.
"""
import json
from pathlib import Path

import torch


def load_model(args):
    root = Path(args.model_name_or_path)
    return {'model': ReplayPolicy(json.loads((root / 'plans.json').read_text()),
                                  json.loads((root / 'observation_contract.json').read_text()))}


class ReplayPolicy:
    def __init__(self, plans, contract):
        self.plans = plans
        self.contract = contract

    def select_action(self, obs):
        assert tuple(obs['image'].shape[1:]) == tuple(self.contract['image_shape'])
        if self.contract['depth_shape'] is not None:
            assert tuple(obs['depth'].shape[1:]) == tuple(self.contract['depth_shape'])
            assert torch.isfinite(obs['depth']).all()
        if self.contract['action_mode'] == 'candidate_viewpoint':
            assert all('candidates' in row for row in obs['navigation'])
        actions = []
        for instruction, timestamp in zip(obs['raw_lang'], obs['timestamp']):
            step = int(timestamp)
            plan = self.plans[instruction]
            if not 0 <= step < len(plan):
                raise ValueError('Replay plan exhausted or invalid timestamp')
            actions.append(plan[step])
        return torch.tensor(actions, dtype=torch.float32).reshape(-1, 1, 1)
