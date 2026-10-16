"""Official Bench2Drive demonstrations with lazy downloads and data-only indices."""
import gzip
import json
import os
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from vlastudio.utils.driving import action_table, compose_images, driving_observation, local_points
from vlastudio.utils.driving_assets import B2D_DATA_REVISION, cache_root, extract, hf_asset


class Bench2DriveDataset(Dataset):
    ctrl_space = 'vehicle'
    ctrl_type = 'control'

    def __init__(self, dataset_dir=None, max_trajectories=10, download=True,
                 seed=20261016, split='train', chunk_size=1, download_workers=4):
        if split != 'train':
            raise ValueError('Bench2DriveDataset provides training demonstrations; evaluate using the closed-loop environment')
        if chunk_size != 1:
            raise ValueError('The TCP preset uses one current control label plus four future waypoint labels')
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError('max_trajectories must be positive or null')
        if download_workers < 1:
            raise ValueError('download_workers must be positive')
        data_cache = Path(os.environ.get('VLASTUDIO_DATA_CACHE_DIR', str(cache_root()/'datasets')))
        root = Path(dataset_dir).expanduser() if dataset_dir else data_cache/'bench2drive-v004'
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.routes = []
        local = sorted(p for p in root.iterdir() if p.is_dir() and (p/'anno').is_dir())
        if local and dataset_dir:
            self.routes = local[:max_trajectories]
        else:
            from filelock import FileLock
            manifest = root/'official-files.json'
            with FileLock(str(manifest)+'.lock'):
                if not manifest.is_file():
                    if not download:
                        raise FileNotFoundError(f'Official Bench2Drive manifest is not cached: {manifest}')
                    from huggingface_hub import HfApi
                    files = []
                    for item in HfApi().list_repo_tree('rethinklab/Bench2Drive-V0.0.4', repo_type='dataset', revision=B2D_DATA_REVISION):
                        if item.path.endswith('.tar.gz'):
                            files.append(item.path)
                    temporary = manifest.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'revision': B2D_DATA_REVISION, 'files': sorted(files)}, indent=2)+'\n')
                    temporary.replace(manifest)
                payload = json.loads(manifest.read_text())
            if payload['revision'] != B2D_DATA_REVISION:
                raise ValueError('Bench2Drive data revision differs from the pinned release')
            groups = {}
            for name in payload['files']:
                groups.setdefault(name.split('_Town')[0], []).append(name)
            rng = random.Random(seed)
            group_names = sorted(groups)
            rng.shuffle(group_names)
            for values in groups.values():
                rng.shuffle(values)
            chosen = []
            while any(groups.values()):
                for name in group_names:
                    if groups[name]:
                        chosen.append(groups[name].pop())
            chosen = chosen[:max_trajectories]
            def materialize(name):
                target = root/Path(name).name.removesuffix('.tar.gz')
                if not target.joinpath('.vlastudio-asset.json').is_file():
                    archive = hf_asset('rethinklab/Bench2Drive-V0.0.4', name, B2D_DATA_REVISION,
                                       dataset=True, download=download)
                    extract(archive, target, strip_root=True)
                return target
            with ThreadPoolExecutor(max_workers=download_workers) as workers:
                self.routes = list(workers.map(materialize, chosen))
        self.frames = []
        self.data = []
        for route_index, route in enumerate(self.routes):
            # A terminal sensor observation may have no subsequent expert action.
            # Match the authors' supervised horizon, indexed by expert assessments.
            assessments = sorted(path for path in (route/'expert_assessment').glob('*.npz') if int(path.stem) >= 0)
            annotations = [route/'anno'/(path.stem+'.json.gz') for path in assessments]
            frame_numbers = [int(path.stem) for path in assessments]
            if frame_numbers != list(range(len(frame_numbers))):
                raise ValueError(f'Non-contiguous expert frames in {route}')
            if len(annotations) <= 25:
                raise ValueError(f'Bench2Drive route has no complete future-label windows: {route}')
            values, experts = [], []
            for path in annotations:
                row = json.loads(gzip.decompress(path.read_bytes()))
                values.append([row[k] for k in ['x', 'y', 'theta', 'x_target', 'y_target', 'speed', 'next_command']])
                expert_path = route/'expert_assessment'/path.name.removesuffix('.json.gz')
                with np.load(expert_path.with_suffix('.npz'), allow_pickle=False) as expert:
                    experts.append(expert['arr_0'])
            values = np.asarray(values, dtype=np.float32)
            experts = np.asarray(experts, dtype=np.float32)
            if experts.ndim != 2 or experts.shape[1] != 1538 or not np.isfinite(experts).all():
                raise ValueError(f'Unexpected expert tensor layout in {route}')
            labels = experts[:, -1]
            if not ((labels == np.floor(labels)) & (labels >= 0) & (labels < len(action_table()))).all():
                raise ValueError(f'Invalid expert action classes in {route}')
            self.data.append((values, experts, [p.name.removesuffix('.json.gz') for p in annotations]))
            self.frames.extend((route_index, i) for i in range(len(values)-25))
        if not self.frames:
            raise ValueError('No Bench2Drive training frames selected')

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        route_index, frame = self.frames[index]
        route = self.routes[route_index]
        values, experts, identifiers = self.data[route_index]
        row = values[frame]
        target = local_points(row[None, 3:5], row[:2], row[2])[0]
        future = [frame+offset for offset in [5, 10, 15, 20]]
        waypoints = local_points(values[future, :2], row[:2], row[2]).astype(np.float32)
        images = []
        for camera in ['rgb_front_left', 'rgb_front', 'rgb_front_right']:
            with Image.open(route/'camera'/camera/(identifiers[frame]+'.jpg')) as image:
                images.append(np.asarray(image.convert('RGB')))
        # Expert discrete actions are the authors' 39 throttle/steer/brake choices.
        action_index = int(experts[frame, -1])
        throttle, steer, brake, reverse = action_table()[action_index]
        return {'image': compose_images(images)[None], **driving_observation(row[5], target, row[6]),
                'action': np.asarray([[steer, throttle, brake, float(reverse)]], dtype=np.float32),
                'is_pad': np.zeros(1, dtype=bool),
                'waypoints': waypoints, 'action_index': action_index,
                'future_action_index': experts[future, -1].astype(np.int64),
                'expert_features': experts[frame, :-2].copy(), 'expert_value': experts[frame, -2].copy(),
                'future_features': experts[future, :-2].copy(),
                'episode_id': route.name, 'frame_id': identifiers[frame]}
