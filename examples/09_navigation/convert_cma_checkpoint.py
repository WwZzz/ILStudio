"""Convert an author-published legacy CMA checkpoint to tensor/plain-data format.

Uses PyTorch's restricted weights-only loader, with the legacy Habitat config
mapped to a known data-only dictionary class. No unrestricted pickle load.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from yacs.config import CfgNode


class LegacyConfig(CfgNode):
    pass


LegacyConfig.__module__ = 'habitat.config.default'
LegacyConfig.__name__ = 'Config'


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unexpected config value: {type(value).__name__}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != args.sha256:
        raise ValueError('Source checkpoint hash mismatch')
    torch.serialization.add_safe_globals([LegacyConfig, set])
    try:
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
    config = plain(checkpoint['config'])
    model_config = copy.deepcopy(config.get('MODEL', config.get('VLN')))
    if model_config is None:
        raise ValueError('Missing model configuration')
    if 'RGB_ENCODER' not in model_config:
        model_config['RGB_ENCODER'] = model_config.pop('VISUAL_ENCODER')
    for field in ['RGB_ENCODER', 'DEPTH_ENCODER']:
        model_config[field]['trainable'] = False
    model_config['DEPTH_ENCODER']['ddppo_checkpoint'] = 'NONE'
    model_config['INSTRUCTION_ENCODER'].update(sensor_uuid='instruction',
        use_pretrained_embeddings=False, final_state_only=False)
    for field in ['normalize_rgb', 'ablate_instruction', 'ablate_rgb', 'ablate_depth']:
        model_config.setdefault(field, False)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': checkpoint['state_dict'], 'model_config': model_config,
                'source_sha256': digest}, output / 'policy.pt')
    (output / 'source_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps({'ctrl_space': 'navigation', 'ctrl_type': 'discrete',
        'chunk_size': 1, 'state_dim': 2, 'action_dim': 1}))
    (output / 'policy_metadata.json').write_text(json.dumps({'policy_module': 'policy.vlnce_cma', 'policy_name': 'vlnce_cma'}))
    (output / 'normalize.json').write_text(json.dumps({'datasets': [{'dataset_id': 'r2r_vlnce',
        'ctrl_space': 'navigation', 'ctrl_type': 'discrete'}], 'state': {'r2r_vlnce': 'identity'},
        'action': {'r2r_vlnce': 'identity'}}))
    print(f'Converted {len(checkpoint["state_dict"])} tensors; original checkpoint SHA256={digest}')


if __name__ == '__main__':
    main()
