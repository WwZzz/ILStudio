"""Navigation contracts; no scene assets or native simulator required."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlastudio.benchmark.base import MetaAction, MetaObs
from vlastudio.benchmark.navigation import NavigationEpisode, discrete_action, evaluate, spl
from vlastudio.benchmark.matterport3d import graph_distances, load_graph, MatterportNavigationEnv


@pytest.mark.parametrize('bad', [-1, 4, 0.5, float('nan'), [1, 2], float('inf')])
def test_discrete_action_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        discrete_action(bad, 4)


def test_action_tags_and_shape():
    assert discrete_action(MetaAction(ctrl_space='navigation', ctrl_type='discrete', action=np.array([2.])), 4) == 2
    with pytest.raises(ValueError):
        discrete_action(MetaAction(action=np.array([2.])), 4)


@pytest.mark.parametrize('stopped,distance,steps,success,done', [
    (True, 2., 0, False, True), (False, 0.1, 0, False, False),
    (True, 0.1, 0, True, True), (False, 0.1, 4, False, True)])
def test_stop_and_timeout_are_not_success(stopped, distance, steps, success, done):
    episode = NavigationEpisode()
    episode.max_steps, episode.success_distance, episode.episode_id = 5, .2, 'e'
    episode.prev_obs = MetaObs(state=np.array([distance, 0.]))
    episode._begin(2.)
    episode.steps = steps
    _, reward, actual_done, info = episode._finish_step(distance, stopped)
    assert info['success'] is success
    assert actual_done is done
    assert reward == float(success)


def test_spl_uses_geodesic_path_length():
    assert spl(True, 4., 8.) == .5
    assert spl(False, 4., 0.) == 0.
    assert spl(True, 0., 0.) == 1.


def test_vector_evaluator_does_not_count_failure_and_freezes_finished_slots():
    class Envs:
        def __init__(self):
            self.index = 0
            self.closed = False
        def __len__(self):
            return 2
        def reset(self):
            return [MetaObs(state=np.zeros(2)), MetaObs(state=np.zeros(2))]
        def step(self, action):
            self.index += 1
            # First slot fails immediately, then tries to report success on later steps.
            return self.reset(), [0, 0], [True, self.index == 2], [
                {'success': self.index > 1, 'spl': 0.}, {'success': self.index == 2, 'spl': .5}]
        def close(self):
            self.closed = True
    env = Envs()
    result = evaluate(SimpleNamespace(max_timesteps=4), env, SimpleNamespace(select_action=lambda: [0, 0]))
    assert result['success'] == [False, True]
    assert result['success_rate'] == .5
    assert result['spl'] == .25
    assert result['horizon'] == [1, 2]
    assert env.closed


def test_r2r_graph_distance_is_not_straight_line(tmp_path):
    nodes = []
    for i, (name, xyz, connected) in enumerate([
        ('a', [0, 0, 0], [False, True, False]),
        ('b', [0, 3, 0], [True, False, True]),
        ('c', [4, 3, 0], [False, True, False])]):
        pose = np.eye(4)
        pose[:3, 3] = xyz
        nodes.append({'image_id': name, 'included': True, 'pose': pose.ravel().tolist(),
                      'unobstructed': connected})
    path = tmp_path / 'graph.json'
    path.write_text(json.dumps(nodes))
    assert graph_distances(load_graph(path), 'c')['a'] == 7.


def test_matterport_turn_increments_rgb_depth_and_no_goal_leakage():
    class Sim:
        def __init__(self):
            self.actions = []
        def getState(self):
            return [SimpleNamespace(heading=0., elevation=0.,
                navigableLocations=[SimpleNamespace()], location=SimpleNamespace(viewpointId='a'),
                rgb=np.array([[[1, 2, 3]]], dtype=np.uint8), depth=np.array([[[4000]]], dtype=np.uint16))]
        def makeAction(self, index, heading, elevation):
            self.actions.append((index, heading, elevation))
    env = MatterportNavigationEnv.__new__(MatterportNavigationEnv)
    env.env = Sim()
    env.episode = {'instruction': 'Turn left and stop'}
    env.episode_id, env.max_steps, env.success_distance = 'a', 10, 3.
    env.distances = {'a': 5.}
    env.graph = {'a': {}}
    env._begin(5.)
    env.prev_obs = env.obs2meta(env.env.getState()[0])
    np.testing.assert_array_equal(env.prev_obs.image[0, :, 0, 0], [3, 2, 1])
    assert env.prev_obs.depth.item() == 1.
    np.testing.assert_array_equal(env.prev_obs.state, [0., 0., 0.])
    env.step(2)
    assert env.env.actions[-1] == ([0], [-1.], [0.])
    _, _, done, info = env.step(0)
    assert done and not info['success']
    count = len(env.env.actions)
    env.step(1)
    assert len(env.env.actions) == count


def test_mlp_checkpoint_and_batched_discrete_actions(tmp_path):
    import torch
    from vlastudio.policy.navigation_mlp import NavigationMLP, load_model
    model = NavigationMLP()
    torch.save({'state_dict': model.state_dict()}, tmp_path / 'policy.pt')
    loaded = load_model(SimpleNamespace(model_name_or_path=str(tmp_path), device='cpu'))['model']
    obs = MetaObs(state=np.array([[2., -.5], [.5, .2]], dtype=np.float32))
    action = loaded.inference(obs)[0]
    assert len(action) == 2
    assert all(0 <= discrete_action(a, 4) < 4 for a in action)
    assert loaded.select_action({'state': torch.from_numpy(obs.state)}).shape == (2, 1, 1)


@pytest.mark.parametrize('timestamp', [0, [0, 1], np.array([0, 1])])
def test_pointnav_policy_through_framework_meta_policy_without_images(timestamp):
    from vlastudio.benchmark.base import MetaPolicy
    from vlastudio.data_utils.normalize import Identity
    from vlastudio.policy.navigation_mlp import NavigationMLP
    normalizer = Identity(ctrl_space='navigation', ctrl_type='discrete')
    policy = MetaPolicy(NavigationMLP(), state_normalizer=normalizer, action_normalizer=normalizer,
                        ctrl_space='navigation', ctrl_type='discrete')
    actions = policy.inference(MetaObs(state=np.array([[2., .5], [1., -.1]], dtype=np.float32),
                                       timestep=timestamp))
    assert len(actions) == 1 and len(actions[0]) == 2
    assert all(0 <= discrete_action(action, 4) < 4 for action in actions[0])


def test_habitat_lab_adapter_preserves_instruction_and_uses_task_success(monkeypatch):
    from vlastudio.benchmark.habitat import HabitatLabEnv
    class Lab:
        def __init__(self, config):
            self.episode_over = False
            self.current_episode = SimpleNamespace(episode_id='r2r-1',
                instruction=SimpleNamespace(instruction_text='Walk through the door.'))
        def reset(self):
            return {'rgb': np.zeros((2, 2, 3), dtype=np.uint8), 'depth': np.ones((2, 2, 1))}
        def step(self, action):
            assert action == 'stop'
            self.episode_over = True
            return self.reset()
        def get_metrics(self):
            return {'success': False, 'spl': 0., 'distance_to_goal': 7.}
        def close(self):
            pass
    calls = []
    monkeypatch.setitem(sys.modules, 'habitat', SimpleNamespace(Env=Lab,
        get_config=lambda **kwargs: calls.append(kwargs)))
    env = HabitatLabEnv(SimpleNamespace(habitat_config='benchmark/nav/vln_r2r.yaml',
                                        habitat_overrides=['habitat.simulator.turn_angle=30']))
    assert calls == [{'config_path': 'benchmark/nav/vln_r2r.yaml', 'overrides': ['habitat.simulator.turn_angle=30']}]
    obs = env.reset()
    assert obs.raw_lang == 'Walk through the door.'
    np.testing.assert_array_equal(obs.state, [0., 0.])
    assert obs.image.shape == (1, 3, 2, 2) and obs.depth.shape == (1, 2, 2)
    _, reward, done, info = env.step(0)
    assert done and not info['success'] and reward == 0.
