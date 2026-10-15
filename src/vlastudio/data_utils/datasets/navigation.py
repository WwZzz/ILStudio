"""Standard train-only VLN datasets with lazy official downloads and replay caches.

Annotations define supervision; simulators render/replay the recorded trajectories.
No learned-policy collection or DAgger is required for these offline datasets.
"""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from vlastudio.utils.navigation_assets import dataset_root, panorama_features, r2r_annotations, sha256, vlnce_annotations


def _selection(episodes, maximum, seed, route_key, scene_key):
    """Select routes from the declared split, covering scenes before repeats."""
    routes = {}
    for episode in episodes:
        routes.setdefault(episode[route_key], []).append(episode)
    pool = list(routes.values())
    random.Random(seed).shuffle(pool)
    chosen, seen = [], set()
    for route in pool:
        if route[0][scene_key] not in seen:
            chosen.append(route)
            seen.add(route[0][scene_key])
    chosen_ids = {r[0][route_key] for r in chosen}
    chosen += [r for r in pool if r[0][route_key] not in chosen_ids]
    if maximum is not None:
        if maximum < 1:
            raise ValueError('max_trajectories must be positive or null')
        chosen = chosen[:maximum]
    return [episode for route in chosen for episode in route]


class _NavigationDataset(Dataset):
    ctrl_space = 'navigation'

    def _cache(self, signature, directory):
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.cache_dir = (Path(directory) if directory else dataset_root()/'replay')/digest
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir/'provenance.json').write_text(json.dumps(signature, indent=2)+'\n')
        self._env = None

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        from filelock import FileLock
        episode = self.episodes[index]
        path = self.cache_dir/f'{episode["episode_id"]}.pt'
        with FileLock(str(path)+'.lock'):
            if path.is_file():
                return torch.load(path, map_location='cpu', weights_only=True)
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            try:
                with torch.random.fork_rng(devices=devices):
                    sample = self._replay(episode)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
            temporary = path.with_suffix('.partial')
            torch.save(sample, temporary)
            temporary.replace(path)
            return sample

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_env'] = state['_extractor'] = None
        if '_features' in state:
            state['_features'] = state['_feature_index'] = None
        return state


class R2RDataset(_NavigationDataset):
    """Official R2R instructions and expert graph trajectories, with cached visuals."""
    ctrl_type = 'candidate_viewpoint'

    def __init__(self, dataset_dir=None, connectivity_path='Matterport3DSimulator/connectivity',
                 feature_store=None, split='train', max_trajectories=None, seed=20261015,
                 cache_dir=None, download=True, chunk_size=1):
        annotations = r2r_annotations(split, dataset_dir, download)
        self.feature_store = panorama_features(feature_store, download)
        raw = json.loads(annotations.read_text())
        expanded = [{**row, 'instruction': instruction, 'episode_id': f'{row["path_id"]}_{i}'}
                    for row in raw for i, instruction in enumerate(row['instructions'])]
        self.episodes = _selection(expanded, max_trajectories, seed, 'path_id', 'scan')
        self.config = dict(episodes=str(annotations), dataset_path='data/v1/scans',
                           connectivity_path=str(Path(connectivity_path).expanduser().resolve()),
                           ctrl_space='navigation', ctrl_type='candidate_viewpoint',
                           action_mode='candidate_viewpoint', candidate_source='simulator',
                           candidate_scope='all_neighbors', candidate_view_alignment='panoramic',
                           rendering_enabled=False, depth_enabled=False, max_timesteps=15,
                           success_distance=3., success_requires_stop=False, image_size=640)
        graphs = {scan: sha256(Path(self.config['connectivity_path'])/f'{scan}_connectivity.json')
                  for scan in {e['scan'] for e in self.episodes}}
        self._cache({'dataset': 'R2R', 'split': split, 'annotations_sha256': sha256(annotations),
                     'graphs': graphs, 'feature_store': str(self.feature_store.resolve()),
                     'feature_index_sha256': sha256(self.feature_store/'index.json'),
                     'feature_size': (self.feature_store/'features.npy').stat().st_size,
                     'feature_mtime_ns': (self.feature_store/'features.npy').stat().st_mtime_ns,
                     'protocol': 'panoramic-reference-15-v3'}, cache_dir)
        self._features = self._feature_index = None

    def _replay(self, episode):
        from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
        if self._env is None:
            self._env = MatterportNavigationEnv(SimpleNamespace(**self.config))
            self._indices = {e['episode_id']: i for i, e in enumerate(self._env.episodes)}
        if self._features is None:
            self._features = np.load(self.feature_store/'features.npy', mmap_mode='r')
            self._feature_index = json.loads((self.feature_store/'index.json').read_text())
        env = self._env
        env.index = self._indices[episode['episode_id']]
        obs = asdict(env.reset())
        candidates, angles, targets, actions, states = [], [], [], [], []
        for step in range(15):
            row = obs['navigation']
            panorama = self._features[self._feature_index[row['scan_id']+'_'+row['viewpoint_id']]]
            features = np.zeros((len(row['candidates'])+1, 2176), dtype=np.float32)
            for i, candidate in enumerate(row['candidates']):
                features[i, :2048] = panorama[candidate['source_view_index']]
                features[i, 2048:] = np.tile([np.sin(candidate['relative_heading']), np.cos(candidate['relative_heading']),
                                            np.sin(candidate['elevation']), np.cos(candidate['elevation'])], 32)
            candidates.append(torch.from_numpy(features))
            heading, elevation = obs['state'][:2]
            states.append(torch.tensor(obs['state'], dtype=torch.float32))
            angles.append(torch.tensor(np.tile([np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)], 32), dtype=torch.float32))
            if row['viewpoint_id'] == episode['path'][-1]:
                action, target = 0, len(row['candidates'])
            else:
                expected = episode['path'][step+1]
                target = next((i for i, candidate in enumerate(row['candidates']) if candidate['viewpoint_id'] == expected), None)
                if target is None:
                    raise RuntimeError(f'Original R2R route edge is missing: {row["viewpoint_id"]} -> {expected}')
                action = row['candidates'][target]['action_id']
            targets.append(target)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                break
        width = max(len(x) for x in candidates)
        features = torch.zeros(len(candidates), width, 2176)
        valid = torch.zeros(len(candidates), width, dtype=torch.bool)
        for i, value in enumerate(candidates):
            features[i, :len(value)], valid[i, :len(value)] = value, True
        return {'candidate_features': features, 'candidate_mask': valid, 'angles': torch.stack(angles),
                'action': torch.tensor(actions, dtype=torch.long)[:, None], 'labels': torch.tensor(targets, dtype=torch.long),
                'state': torch.stack(states), 'is_pad': torch.zeros(len(actions), dtype=torch.bool),
                'raw_lang': episode['instruction'],
                'episode_id': episode['episode_id']}


class VLNCEDataset(_NavigationDataset):
    """Official VLN-CE train demonstrations, replayed once and cached as RGBD features."""
    ctrl_type = 'discrete'

    def __init__(self, dataset_dir=None, scenes_dir='data/scene_datasets', split='train',
                 max_trajectories=None, seed=20261015, cache_dir=None, download=True,
                 feature_checkpoint=None, feature_device='cuda', chunk_size=1):
        if split != 'train':
            raise ValueError('VLNCEDataset supervision uses the official train split; evaluate with the environment')
        annotations = vlnce_annotations(split, dataset_dir, download)
        self.ground_truth = json.loads(gzip.decompress(annotations.with_name('train_gt.json.gz').read_bytes()))
        raw = json.loads(gzip.decompress(annotations.read_bytes()))['episodes']
        self.episodes = _selection(raw, max_trajectories, seed, 'trajectory_id', 'scene_id')
        from vlastudio.utils.navigation_assets import resolve_checkpoint
        self.checkpoint = resolve_checkpoint(feature_checkpoint, 'vlnce_cma', download) if feature_checkpoint else None
        self.feature_device, self._extractor = feature_device, None
        template = Path(__file__).resolve().parents[2]/'configs/env/habitat/habitat_vlnce_cma.yaml'
        self.config = yaml.safe_load(template.read_text())
        self.config['habitat_overrides'] = [x for x in self.config['habitat_overrides'] if not x.startswith('habitat.dataset.')]+[
            f'habitat.dataset.data_path={annotations.resolve()}',
            f'habitat.dataset.scenes_dir={Path(scenes_dir).expanduser().resolve()}', 'habitat.dataset.split=train']
        self._cache({'dataset': 'VLN-CE', 'split': split, 'annotations_sha256': sha256(annotations),
                     'ground_truth_sha256': sha256(annotations.with_name('train_gt.json.gz')),
                     'encoder_checkpoint_sha256': sha256(self.checkpoint/'policy.pt') if self.checkpoint else None,
                     'habitat_config': self.config}, cache_dir)

    def _replay(self, episode):
        from vlastudio.benchmark.habitat import HabitatLabEnv
        from vlastudio.policy.vlnce_cma import CMAPolicy
        if self._env is None:
            self._env = HabitatLabEnv(SimpleNamespace(**self.config))
            self._indices = {str(e.episode_id): i for i, e in enumerate(self._env.env.episodes)}
        if self._extractor is None and self.checkpoint is not None:
            self._extractor = CMAPolicy.from_pretrained(self.checkpoint, self.feature_device)
        env, model = self._env, self._extractor
        env.index = self._indices[str(episode['episode_id'])]
        obs = asdict(env.reset())
        assert str(env.env.current_episode.episode_id) == str(episode['episode_id'])
        initial = float(env.env.get_metrics()['distance_to_goal'])
        rgb, depth, actions, progress = [], [], [], []
        for action in self.ground_truth[str(episode['episode_id'])]['actions']:
            rgb.append(obs['image'][0])
            depth.append(obs['depth'][0, :, :, None])
            actions.append(int(action))
            progress.append((initial-float(env.env.get_metrics()['distance_to_goal']))/max(initial, 1e-8))
            obs, _, done, info = env.step(int(action))
            if done:
                break
        if not info['success']:
            raise RuntimeError(f'Official trajectory replay failed for {episode["episode_id"]}; check scenes and simulator version')
        sample = {'instruction': torch.tensor(episode['instruction']['instruction_tokens']),
                  'action': torch.tensor(actions, dtype=torch.long)[:, None],
                  'state': torch.zeros(len(actions), 2), 'is_pad': torch.zeros(len(actions), dtype=torch.bool),
                  'progress': torch.tensor(progress, dtype=torch.float32),
                  'raw_lang': episode['instruction']['instruction_text'], 'episode_id': str(episode['episode_id'])}
        if model is None:
            return dict(sample, rgb=torch.from_numpy(np.stack(rgb)).permute(0, 2, 3, 1),
                        depth=torch.from_numpy(np.stack(depth)).half())
        rgb_features, depth_features = [], []
        with torch.inference_mode():
            for start in range(0, len(actions), 8):
                images = torch.as_tensor(np.stack(rgb[start:start+8]), device=self.feature_device).float()/255.
                depths = torch.as_tensor(np.stack(depth[start:start+8]), device=self.feature_device)
                rgb_features.append(model.net.rgb_encoder.cnn(images).cpu().half())
                depth_features.append(model.net.depth_encoder.visual_encoder({'depth': depths}).cpu().half())
        return dict(sample, rgb_features=torch.cat(rgb_features), depth_features=torch.cat(depth_features))
