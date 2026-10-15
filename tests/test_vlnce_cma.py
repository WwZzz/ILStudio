"""Recurrent model contracts using frozen visual features, without scene assets."""
import sys
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('gym')
pytest.importorskip('yacs')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from vlastudio.policy.vlnce_cma import CMAPolicy


def test_recurrent_reset_discards_prior_episode_but_keeps_instruction():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = {'INSTRUCTION_ENCODER': {'vocab_size': 2504, 'embedding_size': 50,
        'hidden_size': 128, 'rnn_type': 'LSTM', 'bidirectional': True,
        'final_state_only': False, 'sensor_uuid': 'instruction',
        'use_pretrained_embeddings': False, 'fine_tune_embeddings': False},
        'DEPTH_ENCODER': {'cnn_type': 'VlnResnetDepthEncoder', 'output_size': 128,
            'backbone': 'resnet50', 'ddppo_checkpoint': 'NONE', 'trainable': False},
        'RGB_ENCODER': {'cnn_type': 'TorchVisionResNet50', 'output_size': 256, 'trainable': False},
        'STATE_ENCODER': {'hidden_size': 512, 'rnn_type': 'GRU'},
        'PROGRESS_MONITOR': {'use': True, 'alpha': 1.}, 'normalize_rgb': False,
        'ablate_instruction': False, 'ablate_rgb': False, 'ablate_depth': False}
    model = CMAPolicy(cfg).eval()
    obs = {'instruction': torch.tensor([[2, 3, 4, 0], [7, 8, 9, 0]]),
        'rgb_features': torch.randn(2, 2048, 4, 4),
        'depth_features': torch.randn(2, 128, 4, 4)}
    prior = torch.randn(2, 2, 512)
    with torch.no_grad():
        reset_logits, reset_hidden = model(obs, prior, torch.ones(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.bool))
        clean_logits, clean_hidden = model(obs, torch.zeros_like(prior), torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.bool))
        torch.testing.assert_close(reset_logits, clean_logits)
        torch.testing.assert_close(reset_hidden, clean_hidden)
        changed = dict(obs, instruction=torch.tensor([[7, 8, 9, 0], [2, 3, 4, 0]]))
        changed_logits, _ = model(changed, torch.zeros_like(prior), torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.bool))
        assert not torch.allclose(changed_logits, clean_logits)
