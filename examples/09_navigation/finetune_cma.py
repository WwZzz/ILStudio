"""Train-only supervised continuation of CMA using real simulator observations.

This is a small continuation experiment, not the paper's full augmented DAgger
training schedule. Original checkpoints remain immutable. Frozen CNN features
are cached before the learned spatial embeddings, preserving their gradients.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from vlastudio.benchmark.habitat import HabitatLabEnv
from vlastudio.policy.vlnce_cma import CMAPolicy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Training split environment config')
    p.add_argument('--ground-truth', required=True, help='Official train_gt.json.gz')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trajectories', type=int, default=256)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=5)
    p.add_argument('--unroll', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precollected', help='Verified train-only feature directory from collect_cma_dagger.py')
    a = p.parse_args()
    if min(a.trajectories, a.epochs, a.batch_size, a.unroll) < 1:
        p.error('Counts must be positive')
    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.precollected) if a.precollected else out / 'features'
    cache.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    # Require the caller to identify the official training split explicitly.
    overrides = cfg['habitat_overrides']
    data = next(x.split('=', 1)[1] for x in overrides if x.startswith('habitat.dataset.data_path='))
    if Path(data).name != 'train.json.gz' or Path(a.ground_truth).name != 'train_gt.json.gz':
        raise ValueError('Continuation accepts official train annotations only')
    payload = json.loads(gzip.decompress(Path(data).read_bytes()))
    gt = json.loads(gzip.decompress(Path(a.ground_truth).read_bytes()))
    base_checkpoint = torch.load(Path(a.model) / 'policy.pt', weights_only=True, map_location='cpu')
    model = CMAPolicy.from_pretrained(a.model, a.device)
    # One instruction per route: retain every train scene before random sampling.
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
    # Habitat seeds Python's global RNG during native initialization. Preserve
    # the post-selection state so fresh collection and cached runs train alike.
    training_rng = random.Random()
    training_rng.setstate(random.getstate())
    collection = None
    if a.precollected:
        collection = json.loads((cache.parent/'collection.json').read_text())
        expected = {'source_model_sha256': sha(Path(a.model)/'policy.pt'),
            'train_data_sha256': sha(data), 'config_sha256': sha(a.config),
            'episode_ids': [str(e['episode_id']) for e in chosen]}
        if any(collection.get(key) != value for key, value in expected.items()):
            raise ValueError('Precollected features belong to another model, training split, or episode selection')
    signature = {'source_model_sha256': sha(Path(a.model)/'policy.pt'),
        'train_data_sha256': sha(data), 'ground_truth_sha256': sha(a.ground_truth),
        'config_sha256': sha(a.config), 'arguments': vars(a),
        'episode_ids': [str(e['episode_id']) for e in chosen],
        'scene_count': len({e['scene_id'] for e in chosen}),
        'method': collection['method'] if collection else 'teacher-forced train-only CMA continuation with inflection weighting and progress loss'}
    manifest = out / 'training_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Output belongs to another training protocol')
    manifest.write_text(json.dumps(signature, indent=2)+'\n')
    env = None
    try:
        for n, episode in enumerate(sorted(chosen, key=lambda e: e['scene_id'])):
            target = cache / f"{episode['episode_id']}.pt"
            if target.exists():
                continue
            if a.precollected:
                raise ValueError(f'Precollected training episode is missing: {episode["episode_id"]}')
            if env is None:
                env = HabitatLabEnv(SimpleNamespace(**cfg))
                # Habitat reorders dataset.episodes; file offsets are not native offsets.
                by_id = {str(e.episode_id): i for i, e in enumerate(env.env.episodes)}
            env.index = by_id[str(episode['episode_id'])]
            obs = asdict(env.reset())
            if str(env.env.current_episode.episode_id) != str(episode['episode_id']):
                raise RuntimeError('Training episode identity mismatch')
            rgb, depth, actions, progress = [], [], [], []
            original_actions = gt[str(episode['episode_id'])]['actions']
            initial_distance = float(env.env.get_metrics()['distance_to_goal'])
            for action in original_actions:
                rgb.append(obs['image'][0].transpose(1, 2, 0))
                depth.append(obs['depth'][0, :, :, None])
                actions.append(int(action))
                distance = float(env.env.get_metrics()['distance_to_goal'])
                progress.append((initial_distance-distance)/max(initial_distance, 1e-8))
                obs, _, done, info = env.step(int(action))
                if done:
                    break
            rgb_features, depth_features = [], []
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(actions), 8):
                    r = torch.as_tensor(np.stack(rgb[start:start+8]), device=a.device).permute(0, 3, 1, 2).float()/255.
                    d = torch.as_tensor(np.stack(depth[start:start+8]), device=a.device)
                    rgb_features.append(model.net.rgb_encoder.cnn(r).cpu().half())
                    depth_features.append(model.net.depth_encoder.visual_encoder({'depth': d}).cpu().half())
            torch.save({'rgb_features': torch.cat(rgb_features), 'depth_features': torch.cat(depth_features),
                'instruction': torch.tensor(episode['instruction']['instruction_tokens']),
                'actions': torch.tensor(actions), 'progress': torch.tensor(progress),
                'episode_id': str(episode['episode_id']), 'replay_success': bool(info['success']),
                'replay_navigation_error': float(info['distance_to_goal'])}, target)
            print(json.dumps({'collected': n+1, 'total': len(chosen), 'steps': len(actions),
                'replay_success': bool(info['success'])}), flush=True)
    finally:
        if env is not None:
            env.close()
    paths = [cache/f"{e['episode_id']}.pt" for e in chosen]
    replay = [torch.load(x, weights_only=True) for x in paths]
    replay_success = float(np.mean([x['replay_success'] for x in replay]))
    if not a.precollected and replay_success < .95:
        raise RuntimeError(f'Official training trajectory replay success {replay_success:.3f}; investigate simulator alignment before learning')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    history, updates = [], 0
    for epoch in range(a.epochs):
        model.train()
        model.net.rgb_encoder.cnn.eval()
        model.net.depth_encoder.visual_encoder.eval()
        training_rng.shuffle(replay)
        for offset in range(0, len(replay), a.batch_size):
            rows = replay[offset:offset+a.batch_size]
            batch = len(rows)
            length = max(len(x['actions']) for x in rows)
            hidden = model.initial_state(batch)
            for start in range(0, length, a.unroll):
                end = min(length, start+a.unroll)
                valid = torch.zeros(end-start, batch, device=a.device, dtype=torch.bool)
                observations = {}
                for key in ['rgb_features', 'depth_features']:
                    shape = rows[0][key].shape[1:]
                    tensor = torch.zeros((end-start, batch, *shape), device=a.device)
                    for i, row in enumerate(rows):
                        n = max(0, min(end, len(row['actions']))-start)
                        tensor[:n, i] = row[key][start:start+n].to(a.device).float()
                    observations[key] = tensor.flatten(0, 1)
                tokens = torch.stack([x['instruction'] for x in rows]).to(a.device)
                observations['instruction'] = tokens.unsqueeze(0).expand(end-start, -1, -1).flatten(0, 1)
                labels = torch.zeros(end-start, batch, device=a.device, dtype=torch.long)
                previous = torch.zeros_like(labels)
                weights = torch.ones_like(labels, dtype=torch.float32)
                target_progress = torch.zeros_like(weights)
                for i, row in enumerate(rows):
                    n = max(0, min(end, len(row['actions']))-start)
                    labels[:n, i] = row['actions'][start:start+n].to(a.device)
                    target_progress[:n, i] = row['progress'][start:start+n].to(a.device)
                    teacher_prev = torch.cat([torch.zeros(1, dtype=torch.long), row['actions'][:-1]])
                    prev = row.get('previous_actions', teacher_prev)
                    previous[:n, i] = prev[start:start+n].to(a.device)
                    valid[:n, i] = True
                    weights[:n, i] = torch.where(labels[:n, i] != teacher_prev[start:start+n].to(a.device), 3.2, 1.)
                masks = torch.ones_like(valid)
                if start == 0:
                    masks[0] = False
                features, next_hidden = model.net(observations, hidden, previous.flatten(0, 1)[:, None], masks.flatten(0, 1)[:, None])
                logits = model.action_distribution.linear(features)
                ce = F.cross_entropy(logits, labels.flatten(), reduction='none').view_as(weights)
                pm = F.mse_loss(torch.tanh(model.net.progress_monitor(features)).reshape_as(weights), target_progress, reduction='none')
                normalizer = (valid*weights).sum().clamp_min(1)
                action_loss = (ce*valid*weights).sum()/normalizer
                progress_loss = (pm*valid).sum()/valid.sum().clamp_min(1)
                loss = action_loss+progress_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                hidden = next_hidden.detach()
                updates += 1
                history.append({'epoch': epoch, 'update': updates, 'loss': float(loss.detach()),
                    'action_loss': float(action_loss.detach()), 'progress_loss': float(progress_loss.detach()),
                    'gradient_norm': float(grad_norm)})
            if offset % 25 == 0:
                print(json.dumps(history[-1]), flush=True)
        checkpoint_dir = out / f'epoch-{epoch+1}'
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'model_config': base_checkpoint['model_config'],
                    'training': {'epoch': epoch+1, 'updates': updates, 'source_sha256': signature['source_model_sha256']}}, checkpoint_dir/'policy.pt')
        torch.save(optimizer.state_dict(), checkpoint_dir/'optimizer.pt')
        for name in ['config.json', 'normalize.json', 'policy_metadata.json']:
            shutil.copyfile(Path(a.model)/name, checkpoint_dir/name)
        (out/'losses.json').write_text(json.dumps(history, indent=2)+'\n')
    delta = sum(float((p.detach().cpu()-before[n]).square().sum()) for n, p in model.named_parameters() if n in before)**.5
    if not delta > 0:
        raise RuntimeError('No learned parameter changed')
    (out/'training_result.json').write_text(json.dumps({'updates': updates, 'parameter_delta_l2': delta,
        'first_loss': history[0]['loss'], 'last_loss': history[-1]['loss'], 'trajectory_count': len(replay),
        ('collection_success_rate' if a.precollected else 'replay_success_rate'): replay_success, 'final_checkpoint': str(checkpoint_dir),
        'final_model_sha256': sha(checkpoint_dir/'policy.pt')}, indent=2)+'\n')


if __name__ == '__main__':
    main()
