"""PREVALENT Recurrent VLN-BERT with the published R2R input/action protocol."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .vendor.model import VLNBERT


def angle_features(heading, elevation):
    return np.tile([np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)], 32).astype(np.float32)


class RecurrentVLNBERT(nn.Module):
    def __init__(self, feature_store, vocabulary):
        super().__init__()
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer(vocab_file=str(vocabulary), do_lower_case=True)
        store = Path(feature_store)
        self.feature_index = json.loads((store/'index.json').read_text())
        self.features = np.load(store/'features.npy', mmap_mode='r')
        self.navigator = VLNBERT(feature_size=2176)
        self.reset()

    def reset(self):
        self.language = None
        self.language_mask = None
        self.hidden = None

    def encode_instructions(self, instructions):
        encoded = []
        for text in instructions:
            tokens = self.tokenizer.tokenize(text)[:78]
            tokens = ['[CLS]']+tokens+['[SEP]']
            tokens += ['[PAD]']*(80-len(tokens))
            encoded.append(self.tokenizer.convert_tokens_to_ids(tokens))
        ids = torch.tensor(encoded, device=next(self.parameters()).device, dtype=torch.long)
        mask = ids != 0
        hidden, language = self.navigator('language', sentence=ids, attention_mask=mask.long(), lang_mask=mask.long())
        return hidden, language, mask

    def candidate_inputs(self, navigation, states):
        device = next(self.parameters()).device
        lengths = [len(row['candidates'])+1 for row in navigation]
        features = np.zeros((len(navigation), max(lengths), 2176), dtype=np.float32)
        actions = []
        for i, row in enumerate(navigation):
            if row.get('candidate_view_alignment') != 'panoramic':
                raise ValueError('Recurrent VLN-BERT requires candidate_view_alignment=panoramic')
            key = row['scan_id']+'_'+row['viewpoint_id']
            # Only the panorama at the currently observed viewpoint is read.
            panorama = self.features[self.feature_index[key]]
            for j, candidate in enumerate(row['candidates']):
                features[i, j, :2048] = panorama[candidate['source_view_index']]
                features[i, j, 2048:] = angle_features(candidate['relative_heading'], candidate['elevation'])
            actions.append([c['action_id'] for c in row['candidates']]+[0])
        state_array = np.asarray(states.detach().cpu() if isinstance(states, torch.Tensor) else states)
        heading_features = np.stack([angle_features(float(s[0]), float(s[1])) for s in state_array])
        mask = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        return torch.from_numpy(features).to(device), torch.from_numpy(heading_features).to(device), mask, actions

    def navigate(self, language, hidden, language_mask, navigation, states):
        candidates, angles, valid, action_ids = self.candidate_inputs(navigation, states)
        sentence = torch.cat([hidden[:, None], language[:, 1:]], dim=1)
        self.navigator.vln_bert.config.directions = candidates.shape[1]
        hidden, logits = self.navigator('visual', sentence=sentence,
            attention_mask=torch.cat([language_mask, valid], dim=1).long(),
            lang_mask=language_mask.long(), vis_mask=valid.long(), action_feats=angles, cand_feats=candidates)
        return hidden, logits.masked_fill(~valid, -torch.inf), action_ids

    @torch.inference_mode()
    def select_action(self, obs):
        rows = obs['navigation']
        reset = torch.as_tensor(obs['timestamp']).reshape(-1) == 0
        device = next(self.parameters()).device
        if self.hidden is None or self.hidden.shape[0] != len(rows):
            self.hidden, self.language, self.language_mask = self.encode_instructions(obs['raw_lang'])
        elif reset.any():
            indices = reset.nonzero().flatten().tolist()
            hidden, language, mask = self.encode_instructions([obs['raw_lang'][i] for i in indices])
            self.hidden[indices], self.language[indices], self.language_mask[indices] = hidden, language, mask
        self.hidden, logits, action_ids = self.navigate(self.language, self.hidden,
            self.language_mask, rows, obs['state'])
        selected = logits.argmax(-1).tolist()
        return torch.tensor([ids[index] for ids, index in zip(action_ids, selected)],
                            device=device, dtype=torch.float32).reshape(-1, 1, 1)

    @classmethod
    def from_pretrained(cls, path, device='cpu', feature_store=None):
        path = Path(path)
        config = json.loads((path/'config.json').read_text())
        model = cls(feature_store or config['feature_store'], path/'vocab.txt')
        checkpoint = torch.load(path/'policy.pt', map_location='cpu', weights_only=True)
        model.navigator.load_state_dict(checkpoint['state_dict'], strict=True)
        return model.to(device).eval()


def load_model(args):
    return {'model': RecurrentVLNBERT.from_pretrained(args.model_name_or_path,
        getattr(args, 'device', 'cpu'), getattr(args, 'feature_store', None))}
