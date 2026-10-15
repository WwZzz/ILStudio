"""Verify and extract the official navigator tensors with restricted loading."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from evaluate_vlnbert import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--vocabulary', required=True)
    p.add_argument('--feature-store', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if digest(a.source) != a.sha256:
        raise ValueError('Source checkpoint checksum mismatch')
    if digest(a.vocabulary) != '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3':
        raise ValueError('BERT uncased vocabulary checksum mismatch')
    source = torch.load(a.source, weights_only=True, map_location='cpu')
    state = source['vln_bert']['state_dict']
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError('Navigator contains non-tensor state')
    from vlastudio.policy.recurrent_vlnbert.vendor.model import VLNBERT
    model = VLNBERT(feature_size=2176)
    model.load_state_dict(state, strict=True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'source_sha256': a.sha256,
                'source_epoch': source['vln_bert'].get('epoch')}, out/'policy.pt')
    (out/'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'candidate_viewpoint',
        'state_dim': 3, 'action_dim': 1, 'chunk_size': 1, 'feature_store': str(Path(a.feature_store).resolve())}, indent=2)+'\n')
    (out/'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.recurrent_vlnbert', 'policy_name': 'recurrent_vlnbert'}))
    (out/'normalize.json').write_text(json.dumps({'datasets': [{'dataset_id': 'r2r',
        'ctrl_space': 'navigation', 'ctrl_type': 'candidate_viewpoint'}],
        'state': {'r2r': 'identity'}, 'action': {'r2r': 'identity'}}))
    shutil.copyfile(a.vocabulary, out/'vocab.txt')
    print('Strictly converted official navigator', len(state), 'tensors', flush=True)


if __name__ == '__main__':
    main()
