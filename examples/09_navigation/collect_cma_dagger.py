"""Collect train-only DAgger supervision on actual learner-visited states.

The privileged follower supplies training labels only. Cached features are the
same frozen CNN outputs as ordinary CMA inference, before spatial embeddings.
"""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from vlastudio.benchmark.habitat import HabitatLabEnv
from vlastudio.policy.vlnce_cma import CMAPolicy
from finetune_cma import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trajectories', type=int, default=1000)
    p.add_argument('--teacher-probability', type=float, default=.75)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if not 0 <= a.teacher_probability <= 1 or a.trajectories < 1:
        p.error('Invalid teacher probability or trajectory count')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    cfg = yaml.safe_load(Path(a.config).read_text())
    data = next(x.split('=', 1)[1] for x in cfg['habitat_overrides'] if x.startswith('habitat.dataset.data_path='))
    if Path(data).name != 'train.json.gz':
        raise ValueError('DAgger collection accepts original train annotations only')
    payload = json.loads(gzip.decompress(Path(data).read_bytes()))
    unique = {}
    for episode in payload['episodes']:
        unique.setdefault(episode['trajectory_id'], episode)
    pool = list(unique.values())
    random.shuffle(pool)
    scenes, chosen = set(), []
    for episode in pool:
        if episode['scene_id'] not in scenes:
            scenes.add(episode['scene_id'])
            chosen.append(episode)
    chosen += [e for e in pool if e not in chosen][:max(0, a.trajectories-len(chosen))]
    chosen = chosen[:a.trajectories]
    out = Path(a.output)
    features = out/'features'
    features.mkdir(parents=True, exist_ok=True)
    manifest = {'source_model_sha256': sha(Path(a.model)/'policy.pt'), 'train_data_sha256': sha(data),
        'config_sha256': sha(a.config), 'episode_ids': [str(e['episode_id']) for e in chosen],
        'method': 'Train-only DAgger continuation; 0.5m shortest-path follower labels at learner-visited states',
        'arguments': vars(a), 'scene_count': len({e['scene_id'] for e in chosen})}
    path = out/'collection.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing collection belongs to another protocol')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    model = CMAPolicy.from_pretrained(a.model, a.device)
    if model.net.model_config.normalize_rgb:
        raise ValueError('This collector expects the published CMA normalize_rgb=false checkpoint')
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    env = HabitatLabEnv(SimpleNamespace(**cfg))
    native_indices = {str(e.episode_id): i for i, e in enumerate(env.env.episodes)}
    successes, collected, steps = 0, 0, 0
    try:
        for episode in sorted(chosen, key=lambda e: e['scene_id']):
            target = features/f"{episode['episode_id']}.pt"
            if target.exists():
                cached = torch.load(target, weights_only=True)
                successes += cached['replay_success']
                collected += 1
                steps += len(cached['actions'])
                continue
            env.index = native_indices[str(episode['episode_id'])]
            obs = asdict(env.reset())
            assert str(env.env.current_episode.episode_id) == str(episode['episode_id'])
            follower = ShortestPathFollower(env.env.sim, goal_radius=.5, return_one_hot=False)
            initial_distance = float(env.env.get_metrics()['distance_to_goal'])
            rgb_features, depth_features, labels, previous_actions, executed, progress = [], [], [], [], [], []
            hidden = model.initial_state(1)
            previous = 0
            episode_rng = random.Random(f'{a.seed}:{episode["episode_id"]}')
            for step in range(int(cfg.get('max_timesteps', 500))):
                distance = float(env.env.get_metrics()['distance_to_goal'])
                label = follower.get_next_action(env.env.current_episode.goals[0].position)
                if label is None:
                    if distance >= .5:
                        raise RuntimeError('Follower could not provide a reachable training label')
                    label = 0
                label = int(label)
                with torch.inference_mode():
                    rgb = torch.as_tensor(obs['image'][0], device=a.device).float()[None]/255.
                    depth = torch.as_tensor(obs['depth'][0], device=a.device)[None, :, :, None]
                    rf = model.net.rgb_encoder.cnn(rgb)
                    df = model.net.depth_encoder.visual_encoder({'depth': depth})
                    inputs = {'rgb_features': rf, 'depth_features': df,
                        'instruction': torch.tensor([episode['instruction']['instruction_tokens']], device=a.device)}
                    logits, hidden = model(inputs, hidden,
                        torch.tensor([[previous]], device=a.device),
                        torch.tensor([[step > 0]], device=a.device))
                    prediction = int(logits.argmax(-1).item())
                action = label if episode_rng.random() < a.teacher_probability else prediction
                rgb_features.append(rf.cpu().half())
                depth_features.append(df.cpu().half())
                labels.append(label)
                previous_actions.append(previous)
                executed.append(action)
                progress.append((initial_distance-distance)/max(initial_distance, 1e-8))
                obs, _, done, info = env.step(action)
                previous = action
                if done:
                    break
            cached = {'rgb_features': torch.cat(rgb_features), 'depth_features': torch.cat(depth_features),
                'instruction': torch.tensor(episode['instruction']['instruction_tokens']),
                'actions': torch.tensor(labels), 'previous_actions': torch.tensor(previous_actions),
                'executed_actions': torch.tensor(executed), 'progress': torch.tensor(progress),
                'episode_id': str(episode['episode_id']), 'replay_success': bool(info['success']),
                'replay_navigation_error': float(info['distance_to_goal'])}
            temporary = target.with_suffix('.pt.partial')
            torch.save(cached, temporary)
            temporary.replace(target)
            successes += int(info['success'])
            collected += 1
            steps += len(labels)
            print(json.dumps({'collected': collected, 'total': len(chosen), 'steps': len(labels),
                'collection_success_rate': successes/collected}), flush=True)
    finally:
        env.close()
    result = {'trajectory_count': collected, 'supervised_steps': steps, 'collection_success_rate': successes/collected}
    (out/'collection_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
