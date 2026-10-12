"""CPU checks for the actual default installation; no weights or datasets downloaded."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_default_install_loads_models_data_and_simulator():
    import vlastudio as vla
    from vlastudio.entrypoints import train, eval_sim, start_policy_server
    from vlastudio.data_utils.datasets import AlohaSimDataset
    from vlastudio.policy.act import get_trainer_class
    from vlastudio.policy.diffusion_policy import Trainer
    for policy in ('act', 'diffusion_policy'):
        assert vla.load_policy(policy).runtime_options['runtime'] == 'current'
    assert vla.load_dataset().config_path.is_file()
    assert vla.load_env('aloha_transfer').config_path.is_file()
    assert get_trainer_class() is not None


def test_dp_pooling_coordinates_gradients_and_checkpoint_keys():
    from vlastudio.policy.diffusion_policy.vision import SpatialSoftmax
    pool = SpatialSoftmax((1, 3, 5), num_kp=None)
    # A distribution concentrated in the top-right corner must recover (1, -1).
    features = torch.full((2, 1, 3, 5), -20.0)
    features[:, :, 0, -1] = 20.0
    features.requires_grad_()
    actual = pool(features)
    torch.testing.assert_close(actual, torch.tensor([[[1.0, -1.0]]] * 2))
    actual.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert set(pool.state_dict()) == {'temperature', 'pos_x', 'pos_y'}
    learned = SpatialSoftmax((8, 3, 5), num_kp=4)
    assert set(learned.state_dict()) == {'nets.weight', 'nets.bias', 'temperature', 'pos_x', 'pos_y'}


@pytest.mark.parametrize('policy', ['act', 'diffusion_policy'])
def test_policy_cpu_forward_backward_and_inference(policy, monkeypatch):
    import torchvision.models
    # Random initial ResNet weights keep this test offline. The entire model still runs.
    resnet18 = torchvision.models.resnet18
    def offline_resnet(**kwargs):
        kwargs.pop('pretrained', None)
        kwargs['weights'] = None
        return resnet18(**kwargs)
    monkeypatch.setattr(torchvision.models, 'resnet18', offline_resnet)
    torch.set_num_threads(2)
    if policy == 'act':
        from vlastudio.policy.act import ACTPolicy, ACTPolicyConfig
        model = ACTPolicy(ACTPolicyConfig(camera_names=['primary'], state_dim=4, action_dim=4,
                          hidden_dim=32, dim_feedforward=64, nheads=4,
                          enc_layers=1, dec_layers=1, chunk_size=8))
    else:
        from vlastudio.policy.diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyModel
        from vlastudio.policy.diffusion_policy import vision
        monkeypatch.setattr(vision, 'resnet18', offline_resnet)
        model = DiffusionPolicyModel(DiffusionPolicyConfig(camera_names=['primary'],
                    image_sizes=[(64, 64)], state_dim=4, action_dim=4, chunk_size=8,
                    feature_dimension=8, num_kp=4, num_inference_timesteps=1))
    state = torch.randn(2, 4)
    images = torch.rand(2, 1, 3, 64, 64)
    actions = torch.randn(2, 8, 4)
    padding = torch.zeros(2, 8, dtype=torch.bool)
    loss = model(state, images, actions, padding)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        prediction = model(state, images)
    assert prediction.shape == (2, 8, 4)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize('task', ['sim_transfer_cube_scripted', 'sim_insertion_scripted'])
def test_aloha_physics_loads_and_steps(task):
    from vlastudio.benchmark.aloha.sim_env import make_sim_env
    env = make_sim_env(task)
    try:
        # The task expands the policy's 14 values into 16 physical actuators.
        assert env.physics.model.nu == 16
        env.physics.step()
        assert env.physics.data.time > 0
    finally:
        env.close()
