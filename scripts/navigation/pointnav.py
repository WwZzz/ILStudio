"""Reproducible real-Habitat PointNav BC/DAgger smoke acceptance.

This is GPS/compass PointNav, not language-conditioned R2R. The learned policy
never receives the expert, navmesh, goal world coordinates or test trajectories.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from vlastudio.benchmark.habitat import HabitatNavigationEnv
from vlastudio.policy.navigation_mlp import NavigationMLP, features


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_episodes(config, seed, count):
    env = HabitatNavigationEnv(SimpleNamespace(**{**vars(config), 'seed': seed, 'render': False}))
    try:
        episodes = []
        for _ in range(count):
            env.reset()
            episodes.append(dict(env.episode))
        return episodes
    finally:
        env.close()


def evaluate_policy(env, episodes, policy, record=None):
    rows = []
    frames = []
    for episode_index, episode in enumerate(episodes):
        obs = env.reset(episode)
        actions = []
        while not env.done:
            if record and episode_index == 0 and obs.image is not None:
                frames.append(obs.image[0].transpose(1, 2, 0))
            action = int(policy(obs))
            actions.append(action)
            env.step(action)
            obs = env.prev_obs
        rows.append({**env.last_info, 'actions': actions})
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(record, frames, fps=8)
    return {'success_rate': float(np.mean([r['success'] for r in rows])),
            'spl': float(np.mean([r['spl'] for r in rows])), 'total': len(rows),
            'total_success': sum(r['success'] for r in rows), 'episodes': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', default='results/navigation')
    parser.add_argument('--train-episodes', type=int, default=200)
    parser.add_argument('--val-episodes', type=int, default=40)
    parser.add_argument('--test-episodes', type=int, default=100)
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--validation-seed', type=int, default=100000)
    parser.add_argument('--test-seed', type=int, default=200000)
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if min(args.train_episodes, args.val_episodes, args.test_episodes, args.updates, args.rounds) <= 0:
        parser.error('Episode counts, updates and rounds must be positive')
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    rng = np.random.default_rng(7)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(scene=str(Path(args.scene).resolve()), max_timesteps=80,
                             success_distance=0.2, min_distance=1., max_distance=5.,
                             render=not args.no_render, image_size=128, seed=0)
    # Fix all splits before training; use validation only for model selection.
    splits = {name: make_episodes(config, seed, count) for name, seed, count in [
        ('train', args.train_seed, args.train_episodes), ('validation', args.validation_seed, args.val_episodes),
        ('test', args.test_seed, args.test_episodes)]}
    identities = [{json.dumps((e['start'], e['goal'], e['yaw'])) for e in split} for split in splits.values()]
    assert all(not identities[i].intersection(identities[j]) for i in range(3) for j in range(i))
    write_json(output / 'episodes.json', splits)
    model = NavigationMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    x, y = [], []
    history = []
    expert_failures = 0
    best_score = (-1., -1.)
    best_weights = None
    env = HabitatNavigationEnv(config)
    try:
        for round_index in range(args.rounds):
            for episode_index, episode in enumerate(splits['train']):
                obs = env.reset(episode)
                while not env.done:
                    try:
                        teacher = env.expert_action()
                    except env.hs.errors.GreedyFollowerError:
                        expert_failures += 1
                        break
                    x.append(features(obs.state))
                    y.append(teacher)
                    # DAgger collects teacher labels on states the learner visits.
                    action = teacher if round_index == 0 or rng.random() < 0.3 else model.predict(obs)
                    env.step(action)
                    obs = env.prev_obs
                if (episode_index + 1) % 50 == 0:
                    print(json.dumps({'stage': 'collect', 'round': round_index,
                                      'episodes': episode_index + 1, 'samples': len(y)}), flush=True)
            inputs = torch.as_tensor(np.stack(x))
            labels = torch.tensor(y, dtype=torch.long)
            counts = torch.bincount(labels, minlength=4).clamp_min(1).float()
            criterion = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts).sqrt())
            model.train()
            for update in range(args.updates):
                indices = torch.randint(len(labels), (256,))
                loss = criterion(model(inputs[indices]), labels[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            validation = evaluate_policy(env, splits['validation'], model.predict)
            row = {'round': round_index, 'samples': len(labels), 'loss': float(loss.detach()),
                   'validation_success_rate': validation['success_rate'], 'validation_spl': validation['spl']}
            history.append(row)
            print(json.dumps(row), flush=True)
            score = (validation['success_rate'], validation['spl'])
            if score > best_score:
                best_score = score
                best_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_weights)
        torch.save({'state_dict': model.state_dict(), 'features': 'distance/10,sin(bearing),cos(bearing)',
                    'task': 'gps_compass_pointnav', 'seed': 7}, output / 'policy.pt')
        write_json(output / 'policy_metadata.json', {'policy_module': 'policy.navigation_mlp',
                                                    'policy_name': 'navigation_mlp'})
        write_json(output / 'config.json', {'ctrl_space': 'navigation', 'ctrl_type': 'discrete',
                                          'chunk_size': 1, 'state_dim': 2, 'action_dim': 1})
        write_json(output / 'normalize.json', {
            'datasets': [{'dataset_id': 'pointnav', 'ctrl_space': 'navigation', 'ctrl_type': 'discrete'}],
            'state': {'pointnav': 'identity'}, 'action': {'pointnav': 'identity'}})
        # Verify that the checkpoint, rather than only the in-memory model, can run.
        model = NavigationMLP.from_pretrained(output / 'policy.pt')
        learned = evaluate_policy(env, splits['test'], model.predict,
                                  str(output / 'rollout.mp4') if config.render else None)
        random_rng = np.random.default_rng(19)
        random_result = evaluate_policy(env, splits['test'], lambda _: random_rng.integers(4))
        stop_result = evaluate_policy(env, splits['test'], lambda _: 0)
    finally:
        env.close()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = os.environ.get('VLASTUDIO_SOURCE_REVISION', 'unavailable')
    import habitat_sim
    source_files = [Path(__file__), Path(__file__).resolve().parents[2] / 'src/vlastudio/benchmark/habitat/__init__.py',
                    Path(__file__).resolve().parents[2] / 'src/vlastudio/policy/navigation_mlp/__init__.py']
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_files}
    result = {'task': 'gps_compass_pointnav', 'scene': config.scene,
              'scene_sha256': hashlib.sha256(Path(config.scene).read_bytes()).hexdigest(),
              'protocol': 'same-scene held-out starts/goals; 1-5m geodesic; STOP within 0.2m; 80 steps',
              'not_measured': ['R2R', 'VLN-CE language following', 'unseen-scene generalization'],
              'configuration': vars(config), 'arguments': vars(args), 'revision': revision,
              'python': platform.python_version(), 'torch': torch.__version__,
              'source_sha256': source_hashes,
              'habitat_sim': getattr(habitat_sim, '__version__', 'unknown'),
              'training': history, 'expert_failures': expert_failures,
              'checkpoint_sha256': hashlib.sha256((output / 'policy.pt').read_bytes()).hexdigest(),
              'learned': learned, 'random': random_result, 'stop': stop_result,
              'threshold': args.threshold,
              'accepted': learned['success_rate'] >= args.threshold and learned['success_rate'] > random_result['success_rate']}
    write_json(output / 'evaluation.json', result)
    print(json.dumps({k: result[k] for k in ('accepted', 'threshold', 'checkpoint_sha256')}), flush=True)
    print(json.dumps({'learned_sr': learned['success_rate'], 'learned_spl': learned['spl'],
                      'random_sr': random_result['success_rate'], 'stop_sr': stop_result['success_rate']}), flush=True)
    if not result['accepted']:
        raise SystemExit('Navigation acceptance failed; see evaluation.json')


if __name__ == '__main__':
    main()
