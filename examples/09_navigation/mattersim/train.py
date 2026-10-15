"""Auditable train-only imitation-learning continuation of the published model.

This deliberately small continuation is not the paper's 300k mixed IL/RL
training run. Policy inputs contain no teacher; shortest paths provide labels
and drive teacher-forced simulator actions only in this training program.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
from vlastudio.policy.recurrent_vlnbert import RecurrentVLNBERT
from evaluate import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).resolve().parent/'config/train.yaml'))
    p.add_argument('--model', default='checkpoints/example09/vlnbert-official')
    p.add_argument('--output', default='checkpoints/example09/vlnbert-trained')
    p.add_argument('--trajectories', type=int, default=512)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-7)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if Path(cfg['episodes']).name != 'R2R_train.json':
        raise ValueError('Continuation accepts original R2R_train.json only')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model = RecurrentVLNBERT.from_pretrained(a.model, a.device)
    optimizer = torch.optim.AdamW(model.navigator.parameters(), lr=a.lr, weight_decay=.01)
    envs, history = [], []
    initial = {n: p.detach().cpu().clone() for n, p in model.navigator.named_parameters()}
    try:
        envs = [MatterportNavigationEnv(SimpleNamespace(**cfg)) for _ in range(a.batch_size)]
        pool = list(range(0, len(envs[0].episodes), 3))
        random.shuffle(pool)
        indices = pool[:a.trajectories]
        manifest = {'source_model_sha256': digest(Path(a.model)/'policy.pt'),
            'episodes_sha256': digest(cfg['episodes']), 'config_sha256': digest(a.config),
            'arguments': vars(a), 'episode_ids': [envs[0].episodes[i]['episode_id'] for i in indices],
            'method': 'Teacher-forced train-only IL continuation, full 15-step BPTT; no augmented data or RL'}
        (out/'training_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        for epoch in range(a.epochs):
            model.train()
            random.shuffle(indices)
            for offset in range(0, len(indices), a.batch_size):
                selected = indices[offset:offset+a.batch_size]
                active_envs = envs[:len(selected)]
                observations = []
                for env, index in zip(active_envs, selected):
                    env.index = index
                    observations.append(asdict(env.reset()))
                hidden, language, mask = model.encode_instructions([x['raw_lang'] for x in observations])
                done = [False]*len(selected)
                loss_sum, count = 0., 0
                for _ in range(15):
                    navigation = [x['navigation'] for x in observations]
                    states = np.stack([x['state'] for x in observations])
                    hidden, logits, action_ids = model.navigate(language, hidden, mask, navigation, states)
                    labels = []
                    teachers = []
                    for slot, env in enumerate(active_envs):
                        if done[slot]:
                            labels.append(-100)
                            teachers.append(0)
                            continue
                        current = env.env.getState()[0].location.viewpointId
                        goal = env.episode['path'][-1]
                        if current == goal:
                            action = 0
                        else:
                            candidates = navigation[slot]['candidates']
                            action = min(candidates, key=lambda c: env.distances[c['viewpoint_id']])['action_id']
                        teachers.append(action)
                        labels.append(action_ids[slot].index(action))
                    target = torch.tensor(labels, device=a.device)
                    loss_sum = loss_sum+F.cross_entropy(logits, target, ignore_index=-100, reduction='sum')
                    count += sum(x != -100 for x in labels)
                    for slot, (env, action) in enumerate(zip(active_envs, teachers)):
                        if not done[slot]:
                            observations[slot], _, done[slot], _ = env.step(action)
                    if all(done):
                        break
                loss = loss_sum/max(count, 1)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite continuation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.navigator.parameters(), 40.)
                optimizer.step()
                history.append({'epoch': epoch+1, 'update': len(history)+1,
                    'cross_entropy': float(loss.detach()), 'gradient_norm': float(norm), 'supervised_actions': count})
                if len(history) % 16 == 0:
                    print(json.dumps(history[-1]), flush=True)
            target = out/f'epoch-{epoch+1}'
            target.mkdir(exist_ok=True)
            torch.save({'state_dict': model.navigator.state_dict(), 'training': {'epoch': epoch+1,
                'updates': len(history), 'source_sha256': manifest['source_model_sha256']}}, target/'policy.pt')
            torch.save(optimizer.state_dict(), target/'optimizer.pt')
            for name in ['config.json', 'normalize.json', 'policy_metadata.json', 'vocab.txt']:
                shutil.copyfile(Path(a.model)/name, target/name)
            (out/'losses.json').write_text(json.dumps(history, indent=2)+'\n')
        delta = sum(float((p.detach().cpu()-initial[n]).square().sum()) for n, p in model.navigator.named_parameters())**.5
        if not delta > 0:
            raise RuntimeError('No learned parameters changed')
        (out/'training_result.json').write_text(json.dumps({'updates': len(history), 'parameter_delta_l2': delta,
            'first_loss': history[0]['cross_entropy'], 'last_loss': history[-1]['cross_entropy'],
            'trajectory_count': len(indices), 'final_checkpoint': str(target),
            'final_model_sha256': digest(target/'policy.pt')}, indent=2)+'\n')
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
