"""Driving contracts without downloading assets or starting a simulator."""
import gzip
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from vlastudio.benchmark.base import MetaAction
from vlastudio.benchmark.carla import Bench2DriveEnv, route_metrics
from vlastudio.utils.driving import action_table, local_points, observation_state


def test_official_success_excludes_only_min_speed():
    record = {'route_id': 'r', 'status': 'Completed', 'scores': {'score_composed': 100, 'score_route': 100},
              'infractions': {'min_speed_infractions': ['slow'], 'collisions_vehicle': []}}
    assert route_metrics(record)['success']
    record['infractions']['collisions_vehicle'] = ['collision']
    assert not route_metrics(record)['success']
    record['infractions']['collisions_vehicle'] = []
    record['status'] = 'Failed - Agent crashed'
    assert not route_metrics(record)['success']


def test_vehicle_control_tags_order_ranges_and_reverse():
    env = object.__new__(Bench2DriveEnv)
    action = MetaAction(ctrl_space='vehicle', ctrl_type='control', action=np.array([-.2, .5, 0, 1]))
    assert env.meta2act(action) == [-.2, .5, 0, 1]
    for invalid in [[0, 0, 0], [0, 0, 0, .5], [2, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 1.1, 0], [0, np.nan, 0, 0]]:
        with pytest.raises(ValueError):
            env.meta2act(invalid)
    with pytest.raises(ValueError):
        env.meta2act({'ctrl_space': 'joint', 'action': [0, 0, 0, 0]})


def test_declared_action_order_is_enforced_before_startup():
    with pytest.raises(ValueError, match='action_order'):
        Bench2DriveEnv(SimpleNamespace(action_order=['throttle', 'steer', 'brake', 'reverse']))


def test_driving_coordinates_and_commands():
    np.testing.assert_allclose(local_points([[3, 2]], [1, 2], np.pi/2), [[2, 0]], atol=1e-6)
    np.testing.assert_allclose(local_points([[1, 0]], [0, 0], 0), [[0, 1]], atol=1e-6)
    state = observation_state(3, [2, 1], -1)
    np.testing.assert_equal(state, [3, 2, 1, 0, 0, 0, 1, 0, 0])
    assert len(action_table()) == 39 and action_table()[-1][3]


def test_terminal_observation_and_negative_warmup_have_no_training_actions(tmp_path):
    from PIL import Image
    from vlastudio.data_utils.datasets.bench2drive import Bench2DriveDataset
    route = tmp_path/'Scenario_Town01_Route1_Weather0'
    (route/'anno').mkdir(parents=True)
    (route/'expert_assessment').mkdir()
    for index in range(32):
        row = dict(x=index, y=0, theta=np.pi/2, x_target=30, y_target=0, speed=3, next_command=4)
        (route/'anno'/f'{index:05d}.json.gz').write_bytes(gzip.compress(json.dumps(row).encode()))
    # The terminal frame (31) has no expert; -1 is a sensor warmup action.
    for index in range(-1, 31):
        features = np.full(1538, index, dtype=np.float32)
        features[-1] = 0
        np.savez(route/'expert_assessment'/f'{index:05d}.npz', features)
    for camera, value in [('rgb_front_left', 30), ('rgb_front', 60), ('rgb_front_right', 90)]:
        folder = route/'camera'/camera
        folder.mkdir(parents=True)
        Image.fromarray(np.full((30, 1600, 3), value, dtype=np.uint8)).save(folder/'00000.jpg')
    dataset = Bench2DriveDataset(dataset_dir=tmp_path, download=False)
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample['frame_id'] == '00000'
    np.testing.assert_allclose(sample['waypoints'], [[5, 0], [10, 0], [15, 0], [20, 0]], atol=1e-5)
    np.testing.assert_equal(sample['future_features'][:, 0], [5, 10, 15, 20])
    np.testing.assert_equal(sample['action'], [[0, 0, 1, 0]])
    assert sample['image'].shape == (1, 3, 256, 900)
    assert sample['image'][0, 0, 100, 0] == 30 and sample['image'][0, 0, 100, -1] == 90


def test_batchnorm_freeze_keeps_training_gradients_and_checkpoint_contract(tmp_path, monkeypatch):
    from torch import nn
    from vlastudio.policy.tcp_b2d import TCPConfig, TCPPolicy, upstream
    def network(download):
        return nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1))
    monkeypatch.setattr(upstream, 'network', network)
    model = TCPPolicy(TCPConfig(download=False)).train()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.navigator(torch.randn(2, 3, 4, 4)).square().mean().backward()
    optimizer.step()
    assert torch.equal(original['navigator.0.running_mean'], model.state_dict()['navigator.0.running_mean'])
    assert not torch.equal(original['navigator.1.weight'], model.state_dict()['navigator.1.weight'])
    model.save_pretrained(tmp_path)
    loaded = TCPPolicy.from_pretrained(tmp_path)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    config = json.loads((tmp_path/'config.json').read_text())
    assert (config['action_dim'], config['state_dim']) == (4, 9)
    assert json.loads((tmp_path/'normalize.json').read_text())['action']['bench2drive'] == 'identity'


@pytest.mark.parametrize('name,link', [('../escape', None), ('link', '../../escape')])
def test_asset_extraction_rejects_escaping_members(tmp_path, name, link):
    from vlastudio.utils.driving_assets import extract
    archive = tmp_path/'fixture.tar'
    with tarfile.open(archive, 'w') as output:
        member = tarfile.TarInfo(name)
        if link:
            member.type, member.linkname = tarfile.SYMTYPE, link
            output.addfile(member)
        else:
            member.size = 1
            output.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        extract(archive, tmp_path/'asset')
    assert not (tmp_path/'asset').exists()


def test_carla_recipe_declares_native_installation():
    from vlastudio.configuration import read_config
    from vlastudio.profiles import environment_for
    from vlastudio.env_installers import preflight
    config, path = read_config('carla.bench2drive', 'env')
    profile = environment_for(config, path, check_platform=False)
    assert profile['installers'] == ['carla'] and 'carla' in profile['checks']
    preflight(profile)


def test_native_installer_uses_target_interpreter_and_shared_cache(tmp_path):
    from vlastudio.env_installers import finish
    commands, env = [], {'UV_OFFLINE': '1'}
    finish({'installers': ['carla']}, tmp_path, '/target/bin/python', env, commands.append, pip=lambda *args: None)
    root = str(tmp_path/'driving/runtimes/carla-0.9.15')
    assert commands == [['/target/bin/python', '-m', 'vlastudio.benchmark.carla.install', '--prefix', root, '--offline']]
    assert env['VLASTUDIO_RUNTIME_CARLA_ROOT'] == root


def test_unwrapped_runtime_archive_remains_traversable(tmp_path):
    from vlastudio.utils.driving_assets import extract
    archive = tmp_path/'fixture.tar'
    with tarfile.open(archive, 'w') as output:
        member = tarfile.TarInfo('CarlaUE4.sh')
        member.mode, member.size = 0o755, 1
        output.addfile(member, io.BytesIO(b'x'))
    target = extract(archive, tmp_path/'runtime')
    if __import__('os').name != 'nt':
        assert target.stat().st_mode & 0o777 == 0o755
