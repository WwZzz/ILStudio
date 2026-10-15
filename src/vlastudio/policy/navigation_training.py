"""Navigation losses, batching and checkpoint hooks for the standard trainer."""
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F


def collate_trajectories(samples):
    # Ragged trajectories stay separate; loss functions own sequence padding/state.
    keys = {'rgb_features', 'depth_features', 'rgb', 'depth', 'instruction', 'progress', 'action', 'labels',
            'candidate_features', 'candidate_mask', 'angles', 'raw_lang'}
    return {'episodes': [{k: v for k, v in sample.items() if k in keys} for sample in samples]}


def cma_loss(model, episodes, unroll=32):
    from torch.utils.checkpoint import checkpoint
    device = next(model.parameters()).device
    action_sum = progress_sum = 0.
    weight_sum = count = 0
    for row in episodes:
        labels = row['action'].flatten().to(device=device, dtype=torch.long)
        previous = torch.cat([torch.zeros(1, device=device, dtype=torch.long), labels[:-1]])
        weights = torch.where(labels != previous, 3.2, 1.)
        hidden = model.initial_state(1)
        count += len(labels)
        weight_sum = weight_sum+weights.sum()
        for start in range(0, len(labels), unroll):
            end = min(start+unroll, len(labels))
            keys = ['rgb_features', 'depth_features'] if 'rgb_features' in row else ['rgb', 'depth']
            obs = {key: row[key][start:end].to(device=device, dtype=torch.uint8 if key == 'rgb' else torch.float32)
                   for key in keys}
            obs['instruction'] = row['instruction'].to(device).expand(end-start, -1)
            masks = torch.ones(end-start, 1, device=device, dtype=torch.bool)
            if start == 0:
                masks[0] = False
            features, hidden = checkpoint(model.net, obs, hidden, previous[start:end, None], masks, use_reentrant=False)
            hidden = hidden.detach()
            ce = F.cross_entropy(model.action_distribution.linear(features), labels[start:end], reduction='none')
            prediction = torch.tanh(model.net.progress_monitor(features)).flatten()
            target = row['progress'][start:end].to(device)
            action_sum = action_sum+(ce*weights[start:end]).sum()
            progress_sum = progress_sum+F.mse_loss(prediction, target, reduction='sum')
    return {'loss': action_sum/weight_sum+progress_sum/count}


def vlnbert_loss(model, episodes):
    device = next(model.parameters()).device
    total, count = 0., 0
    for row in episodes:
        hidden, language, mask = model.encode_instructions([row['raw_lang']])
        for features, angles, valid, label in zip(row['candidate_features'], row['angles'], row['candidate_mask'], row['labels']):
            # Upstream feature dropout writes in-place. Each timestep must own
            # storage rather than sharing the trajectory tensor's version counter.
            features = features.to(device=device, dtype=torch.float32)[None].clone()
            valid = valid.to(device)[None]
            sentence = torch.cat([hidden[:, None], language[:, 1:]], dim=1)
            model.navigator.vln_bert.config.directions = features.shape[1]
            hidden, logits = model.navigator('visual', sentence=sentence,
                attention_mask=torch.cat([mask, valid], dim=1).long(), lang_mask=mask.long(),
                vis_mask=valid.long(), action_feats=angles.to(device=device, dtype=torch.float32)[None], cand_feats=features)
            total = total+F.cross_entropy(logits.masked_fill(~valid, -torch.inf), label.to(device).reshape(1).long())
            count += 1
    return {'loss': total/count}


def trainer_class():
    # Trainer/Accelerate remain lazy: model inference does not import training stacks.
    from transformers import Trainer as HFTrainer
    from vlastudio.policy.trainer import BaseTrainer

    class NavigationTrainer(BaseTrainer):
        def _save(self, output_dir=None, state_dict=None):
            target = Path(output_dir or self.args.output_dir)
            self.model.save_pretrained(target, state_dict=state_dict)
            for name in ['normalize.json', 'policy_metadata.json']:
                source = Path(self.args.output_dir)/name
                if source.is_file() and source.resolve() != (target/name).resolve():
                    shutil.copyfile(source, target/name)
            torch.save(self.args, target/'training_args.bin')

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            model = model or self.model
            checkpoint = torch.load(Path(resume_from_checkpoint)/'policy.pt', map_location='cpu', weights_only=True)
            target = model.navigator if hasattr(model, 'navigator') else model
            target.load_state_dict(checkpoint['state_dict'], strict=True)

        def _load_best_model(self):
            self._load_from_checkpoint(self.state.best_model_checkpoint)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
            # Discrete navigation uses supervised loss, not continuous action MSE.
            return HFTrainer.evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)

    return NavigationTrainer


class Trainer:
    def __new__(cls, *args, **kwargs):
        return trainer_class()(*args, **kwargs)
