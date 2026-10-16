"""Policy-independent temporal VLN samples from official demonstrations."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from vlastudio.utils.navigation_assets import dataset_root, r2r_annotations, sha256, vlnce_annotations


def _selection(episodes, maximum, seed, route_key, scene_key):
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


def _plain(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _frame(observation, action):
    navigation = dict((observation.get('reasoning') or {}).get('navigation') or observation.get('navigation') or {})
    # Original annotation tokens are not a public model input contract.
    navigation.pop('instruction_tokens', None)
    sample = {key: observation[key] for key in ['image', 'depth', 'state', 'raw_lang']
              if observation.get(key) is not None}
    sample.update(action=torch.tensor([[action]], dtype=torch.long), is_pad=torch.zeros(1, dtype=torch.bool),
                  navigation=navigation)
    return sample


class _NavigationDataset(Dataset):
    ctrl_space = 'navigation'

    def _sampling(self, sample_mode, chunk_size, image_offsets):
        if sample_mode not in ('episode', 'window'):
            raise ValueError('sample_mode must be episode or window')
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        if sample_mode == 'episode' and image_offsets is not None:
            raise ValueError('Episode samples include every frame; image_offsets is for window samples')
        self.sample_mode, self.chunk_size = sample_mode, chunk_size
        self.image_offsets = list(range(chunk_size)) if image_offsets is None else list(image_offsets)
        if (not self.image_offsets or self.image_offsets[0] != 0 or
                any(not isinstance(i, int) for i in self.image_offsets) or
                any(a >= b for a, b in zip(self.image_offsets, self.image_offsets[1:]))):
            raise ValueError('image_offsets must be strictly increasing integers starting at zero')

    def _cache(self, signature, directory):
        signature = {**signature, 'format': 'raw-navigation-frames-v1'}
        self.source_signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.cache_dir = (Path(directory) if directory else dataset_root()/'replay')/self.source_signature
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir/'provenance.json').write_text(json.dumps(signature, indent=2)+'\n')
        self.cumulative_len = np.cumsum(self.episode_lengths)
        starts = [0]+self.cumulative_len[:-1].tolist()
        self.episode_ranges = list(zip(starts, self.cumulative_len.tolist()))
        self._env = None

    def __len__(self):
        if self.sample_mode == 'episode':
            return len(self.episodes)
        return int(self.cumulative_len[-1]) if len(self.cumulative_len) else 0

    def _materialize(self, episode_index):
        from filelock import FileLock
        episode = self.episodes[episode_index]
        identifier = hashlib.sha256(str(episode['episode_id']).encode()).hexdigest()[:24]
        folder = self.cache_dir/identifier
        with FileLock(str(folder)+'.lock'):
            if (folder/'complete.json').exists():
                manifest = json.loads((folder/'complete.json').read_text())
                if manifest['frames'] != self.episode_lengths[episode_index]:
                    raise ValueError('Raw navigation cache frame count differs from its annotation')
                return folder
            if folder.exists():
                raise ValueError(f'Incomplete raw navigation cache: {folder}')
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
            try:
                with torch.random.fork_rng(devices=devices), tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
                    staging = Path(temporary)/'episode'
                    staging.mkdir()
                    count = 0
                    for count, sample in enumerate(self._replay(episode), 1):
                        sample.update(episode_id=str(episode['episode_id']), frame_id=count-1,
                                      episode_length=self.episode_lengths[episode_index], timestamp=count-1)
                        torch.save(_plain(sample), staging/f'{count-1:06d}.pt')
                    if count != self.episode_lengths[episode_index]:
                        raise RuntimeError('Expert replay ended before its complete recorded trajectory')
                    (staging/'complete.json').write_text(json.dumps({'frames': count}))
                    staging.rename(folder)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        return folder

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = index if self.sample_mode == 'episode' else int(np.searchsorted(self.cumulative_len, index, side='right'))
        start = 0 if self.sample_mode == 'episode' else index-self.episode_ranges[episode][0]
        length = self.episode_lengths[episode]
        action_indices = np.arange(length) if self.sample_mode == 'episode' else start+np.arange(self.chunk_size)
        image_indices = action_indices if self.sample_mode == 'episode' else start+np.array(self.image_offsets)
        folder = self._materialize(episode)
        indices = set(np.minimum(np.concatenate([action_indices, image_indices]), length-1).tolist())
        frames = {i: torch.load(folder/f'{i:06d}.pt', map_location='cpu', weights_only=True) for i in indices}
        actions = [frames[min(int(i), length-1)] for i in action_indices]
        images = [frames[min(int(i), length-1)] for i in image_indices]
        first = actions[0]
        reasoning = dict(image_type='rgb',
                         image_is_pad=torch.from_numpy(image_indices >= length),
                         navigation=[row['navigation'] for row in images],
                         frame_ids=torch.from_numpy(np.minimum(image_indices, length-1)),
                         action_frame_ids=torch.from_numpy(np.minimum(action_indices, length-1)))
        sample = dict(raw_lang=first['raw_lang'], reasoning=reasoning,
                      action=torch.cat([row['action'] for row in actions]),
                      state=torch.stack([row['state'] for row in actions]).float(),
                      is_pad=torch.from_numpy(action_indices >= length),
                      timestamp=start, episode_id=first['episode_id'], __index__=int(index))
        for key in ('image', 'depth'):
            if key in first:
                values = torch.stack([row[key] for row in images], dim=1)  # K,T,..., like HDF5SequenceDataset
                sample[key] = values[:, 0] if len(images) == 1 else values
        if 'supervision' in first:
            reasoning['supervision'] = {key: torch.tensor([row['supervision'][key] for row in actions])
                                     for key in first['supervision']}
        if getattr(self, 'visual_mode', None) == 'features':
            sample['image'] = torch.stack([torch.from_numpy(self.feature_store.read(
                row['navigation']['scan_id'], row['navigation']['viewpoint_id'])) for row in images])
            reasoning.update(image_type='features', visual_feature_source=self.feature_store.signature)
        return sample

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_env'] = None
        return state


class R2RDataset(_NavigationDataset):
    """RGB or published panorama features, text, geometry and expert actions."""
    ctrl_type = 'candidate_viewpoint'

    def __init__(self, dataset_dir=None, connectivity_path='Matterport3DSimulator/connectivity',
                 dataset_path='data/v1/scans', split='train', max_trajectories=None, seed=20261015,
                 cache_dir=None, download=True, chunk_size=1, image_size=256,
                 sample_mode='episode', image_offsets=None, visual_mode='features', feature_store=None, scenes=None):
        self._sampling(sample_mode, chunk_size, image_offsets)
        if visual_mode not in ('rgb', 'features'):
            raise ValueError('visual_mode must be rgb or features')
        self.visual_mode = visual_mode
        if visual_mode == 'features':
            from .navigation_features import PanoramaFeatureStore
            self.feature_store = PanoramaFeatureStore(feature_store, download)
        annotations = r2r_annotations(split, dataset_dir, download)
        raw = json.loads(annotations.read_text())
        if scenes is not None:
            raw = [row for row in raw if row['scan'] in scenes]
        if not raw:
            raise ValueError('No R2R training episodes match the requested scenes')
        expanded = [{**row, 'instruction': instruction, 'episode_id': f'{row["path_id"]}_{i}'}
                    for row in raw for i, instruction in enumerate(row['instructions'])]
        self.episodes = _selection(expanded, max_trajectories, seed, 'path_id', 'scan')
        self.episode_lengths = [len(row['path']) for row in self.episodes]
        self.config = dict(episodes=str(annotations), dataset_path=str(Path(dataset_path).expanduser().resolve()),
                           connectivity_path=str(Path(connectivity_path).expanduser().resolve()),
                           ctrl_space='navigation', ctrl_type='candidate_viewpoint',
                           action_mode='candidate_viewpoint', candidate_source='simulator',
                           candidate_scope='all_neighbors', candidate_view_alignment='panoramic',
                           rendering_enabled=visual_mode == 'rgb', depth_enabled=False,
                           max_timesteps=max(self.episode_lengths, default=1)+1,
                           success_distance=3., success_requires_stop=True, image_size=image_size)
        graphs = {scan: sha256(Path(self.config['connectivity_path'])/f'{scan}_connectivity.json')
                  for scan in {e['scan'] for e in self.episodes}}
        self._cache({'dataset': 'R2R', 'split': split, 'annotations_sha256': sha256(annotations),
                     'graphs': graphs, 'rendering_config': self.config}, cache_dir)

    def _replay(self, episode):
        from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
        if self._env is None:
            self._env = MatterportNavigationEnv(SimpleNamespace(**self.config))
            self._indices = {e['episode_id']: i for i, e in enumerate(self._env.episodes)}
        env = self._env
        env.index = self._indices[episode['episode_id']]
        observation = asdict(env.reset())
        for step, viewpoint in enumerate(episode['path']):
            navigation = observation['reasoning']['navigation']
            if navigation['viewpoint_id'] != viewpoint:
                raise RuntimeError('Replay left the annotated R2R reference path')
            if step == len(episode['path'])-1:
                action = 0
            else:
                expected = episode['path'][step+1]
                action = next((c['action_id'] for c in navigation['candidates'] if c['viewpoint_id'] == expected), None)
                if action is None:
                    raise RuntimeError(f'Original R2R route edge is missing: {viewpoint} -> {expected}')
            yield _frame(observation, action)
            observation, _, done, information = env.step(action)
        if not done or not information['success']:
            raise RuntimeError('Official R2R demonstration replay did not finish successfully')


class VLNCEDataset(_NavigationDataset):
    """Temporal RGB/depth, original text and expert primitive actions; no models."""
    ctrl_type = 'discrete'

    def __init__(self, dataset_dir=None, scenes_dir='data/scene_datasets', split='train',
                 max_trajectories=None, seed=20261015, cache_dir=None, download=True, chunk_size=1,
                 sample_mode='episode', image_offsets=None, scenes=None):
        if split != 'train':
            raise ValueError('VLNCEDataset supervision uses train; evaluate with the environment')
        self._sampling(sample_mode, chunk_size, image_offsets)
        annotations = vlnce_annotations(split, dataset_dir, download)
        self.ground_truth = json.loads(gzip.decompress(annotations.with_name('train_gt.json.gz').read_bytes()))
        raw = json.loads(gzip.decompress(annotations.read_bytes()))['episodes']
        if scenes is not None:
            raw = [row for row in raw if Path(row['scene_id']).stem in scenes]
        if not raw:
            raise ValueError('No VLN-CE training episodes match the requested scenes')
        self.episodes = _selection(raw, max_trajectories, seed, 'trajectory_id', 'scene_id')
        self.episode_lengths = [len(self.ground_truth[str(e['episode_id'])]['actions']) for e in self.episodes]
        template = Path(__file__).resolve().parents[2]/'configs/env/habitat/habitat_vlnce_cma.yaml'
        self.config = yaml.safe_load(template.read_text())
        self.config.update(return_instruction_tokens=False, state_source='none')
        self.config['habitat_overrides'] = [x for x in self.config['habitat_overrides'] if not x.startswith('habitat.dataset.')]+[
            f'habitat.dataset.data_path={annotations.resolve()}',
            f'habitat.dataset.scenes_dir={Path(scenes_dir).expanduser().resolve()}', 'habitat.dataset.split=train',
            'habitat.dataset.content_scenes='+json.dumps(sorted({Path(e['scene_id']).stem for e in self.episodes}))]
        self._cache({'dataset': 'VLN-CE', 'split': split, 'annotations_sha256': sha256(annotations),
                     'ground_truth_sha256': sha256(annotations.with_name('train_gt.json.gz')),
                     'habitat_config': self.config}, cache_dir)

    def _replay(self, episode):
        from vlastudio.benchmark.habitat import HabitatLabEnv
        if self._env is None:
            self._env = HabitatLabEnv(SimpleNamespace(**self.config))
            self._indices = {str(e.episode_id): i for i, e in enumerate(self._env.env.episodes)}
        env = self._env
        env.index = self._indices[str(episode['episode_id'])]
        observation = asdict(env.reset())
        initial = float(env.env.get_metrics()['distance_to_goal'])
        if str(env.env.current_episode.episode_id) != str(episode['episode_id']):
            raise RuntimeError('Habitat reset selected a different episode than the demonstration')
        for action in self.ground_truth[str(episode['episode_id'])]['actions']:
            sample = _frame(observation, int(action))
            sample['supervision'] = {'distance_to_goal': float(env.env.get_metrics()['distance_to_goal']),
                                     'initial_distance_to_goal': initial}
            yield sample
            observation, _, done, information = env.step(int(action))
            if done:
                break
        if not information['success']:
            raise RuntimeError(f'Official trajectory replay failed for {episode["episode_id"]}')
