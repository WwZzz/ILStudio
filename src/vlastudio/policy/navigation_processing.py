"""Adapt dataset-owned sequences; persistence belongs to the data task cache."""
import hashlib
from pathlib import Path

import torch


class NavigationProcessor:
    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        # The shared data cache must invalidate features when the encoder changes.
        digest = hashlib.sha256(Path(__file__).read_bytes())
        if kind == 'cma':
            model.instruction_tokenizer._load()
            import json
            digest.update(json.dumps(model.instruction_tokenizer.vocabulary).encode())
            for encoder in (model.net.rgb_encoder.cnn, model.net.depth_encoder.visual_encoder):
                for name, value in encoder.state_dict().items():
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                digest.update(str([p.requires_grad for p in encoder.parameters()]).encode())
            digest.update(Path(__file__).with_name('vlnce_cma').joinpath('tokenizer.py').read_bytes())
        self.encoding_signature = digest.hexdigest()

    def __call__(self, sample):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError('Navigation model preprocessing requires cache.num_workers=0 / dataloader_num_workers=0')
        valid = ~sample['is_pad'].bool()
        count = int(valid.sum())
        if not count or not valid[:count].all() or valid[count:].any():
            raise ValueError('Navigation windows require contiguous valid actions before padding')
        if len(sample['frame_ids']) < count or not torch.equal(sample['frame_ids'][:count], sample['action_frame_ids'][:count]):
            raise ValueError('These recurrent baselines require an observation for each expert action')
        return self._cma(sample, count) if self.kind == 'cma' else self._vlnbert(sample, count)

    def _cma(self, sample, count):
        model = self.model
        supervision = sample['supervision']
        initial = supervision['initial_distance_to_goal'][:count]
        row = {'raw_lang': sample['raw_lang'], 'episode_id': sample['episode_id'],
               'instruction': torch.tensor(model.instruction_tokenizer(sample['raw_lang'])),
               'action': sample['action'][:count],
               'progress': (initial-supervision['distance_to_goal'][:count])/initial.clamp_min(1e-8)}
        rgb, depth = sample['image'], sample['depth']
        if rgb.ndim == 4:
            rgb, depth = rgb[:, None], depth[:, None]
        rgb, depth = rgb[0, :count], depth[0, :count, ..., None].float()
        encoders = [model.net.rgb_encoder.cnn, model.net.depth_encoder.visual_encoder]
        if any(p.requires_grad for encoder in encoders for p in encoder.parameters()):
            return dict(row, rgb=rgb.permute(0, 2, 3, 1), depth=depth)
        device = next(model.parameters()).device
        rgb_features, depth_features = [], []
        for encoder in encoders:
            encoder.eval()
        with torch.inference_mode():
            for image, distances in zip(rgb, depth):
                rgb_features.append(encoders[0](image[None].to(device).float()/255.)[0].cpu())
                depth_features.append(encoders[1]({'depth': distances[None].to(device)})[0].cpu())
        return dict(row, rgb_features=torch.stack(rgb_features), depth_features=torch.stack(depth_features))

    def _vlnbert(self, sample, count):
        if 'visual_features' not in sample:
            raise ValueError('Recurrent VLN-BERT training requires dataset visual_mode=features')
        features, angles, masks, actions = self.model.candidate_inputs(
            sample['navigation'][:count], sample['state'][:count], sample['visual_features'][:count])
        targets = []
        for action, candidates in zip(sample['action'][:count].flatten().tolist(), actions):
            if action not in candidates:
                raise ValueError('Expert action is absent from the current observable candidates')
            targets.append(candidates.index(action))
        return {'raw_lang': sample['raw_lang'], 'episode_id': sample['episode_id'],
                'candidate_features': features.cpu(), 'candidate_mask': masks.cpu(), 'angles': angles.cpu(),
                'labels': torch.tensor(targets), 'action': sample['action'][:count]}
