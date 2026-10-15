"""Optional simulator affordances and candidate actions through policy transports."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vlastudio.benchmark.base import MetaAction, MetaObs, MetaPolicy, dict2meta
from vlastudio.benchmark.matterport3d import MatterportNavigationEnv
from vlastudio.benchmark.navigation import navigation_options
from vlastudio.benchmark.utils import organize_obs


@pytest.mark.parametrize('options', [
    {'action_mode': 'unknown'}, {'candidate_source': 'policy'},
    {'candidate_scope': 'unknown'}, {'action_mode': 'candidate_viewpoint'},
    {'action_mode': 'candidate_viewpoint', 'candidate_source': 'simulator', 'ctrl_type': 'discrete'},
    {'candidate_source': 'simulator', 'candidate_scope': 'all_neighbors'},
])
def test_invalid_navigation_contract_fails_before_native_import(options):
    with pytest.raises(ValueError):
        MatterportNavigationEnv(SimpleNamespace(**options))


@pytest.mark.parametrize('backend', ['HabitatNavigationEnv', 'HabitatLabEnv'])
def test_habitat_rejects_unsupported_candidate_execution_before_import(backend):
    from vlastudio.benchmark import habitat
    with pytest.raises(ValueError, match='supports primitive'):
        getattr(habitat, backend)(SimpleNamespace(candidate_source='simulator'))


def make_matterport(monkeypatch, tmp_path, **options):
    graph = {'a': ['b', 'c'], 'b': ['a'], 'c': ['a']}
    nodes = []
    for i, name in enumerate(graph):
        pose = np.eye(4)
        pose[0, 3] = i
        nodes.append({'image_id': name, 'included': True, 'pose': pose.ravel().tolist(),
                      'unobstructed': [target in graph[name] for target in graph]})
    (tmp_path / 'scan_connectivity.json').write_text(json.dumps(nodes))
    episodes = tmp_path / 'episodes.json'
    episodes.write_text(json.dumps([{'scan': 'scan', 'path': ['a', 'c'],
                                    'heading': 0., 'path_id': 1, 'instructions': ['Go to c.']}]))

    class Sim:
        def __init__(self):
            self.settings, self.actions = {}, []
            self.current, self.reverse = 'a', False
        def __getattr__(self, key):
            if key.startswith('set'):
                return lambda *args: self.settings.__setitem__(key, args)
            raise AttributeError(key)
        def initialize(self):
            pass
        def newEpisode(self, scan, location, heading, elevation):
            self.current = location[0]
        def getState(self):
            targets = graph[self.current][:]
            if self.settings.get('setRestrictedNavigation', (True,))[0]:
                targets = targets[:1]
            if self.reverse:
                targets.reverse()
            locations = [SimpleNamespace(viewpointId=self.current)] + [
                SimpleNamespace(viewpointId=target, rel_heading=.2 * i,
                                rel_elevation=0., rel_distance=1. + i)
                for i, target in enumerate(targets)]
            return [SimpleNamespace(heading=0., elevation=0., navigableLocations=locations,
                location=SimpleNamespace(viewpointId=self.current),
                rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=np.full((2, 2, 1), 4000, dtype=np.uint16))]
        def makeAction(self, location, heading, elevation):
            self.actions.append((location, heading, elevation))
            self.current = self.getState()[0].navigableLocations[location[0]].viewpointId

    monkeypatch.setitem(sys.modules, 'MatterSim', SimpleNamespace(Simulator=Sim))
    config = SimpleNamespace(episodes=str(episodes), dataset_path=str(tmp_path),
                             connectivity_path=str(tmp_path), **options)
    return MatterportNavigationEnv(config)


def test_candidates_are_opt_in_and_do_not_choose_an_action(monkeypatch, tmp_path):
    env = make_matterport(monkeypatch, tmp_path)
    obs = env.reset()
    assert obs.navigation['candidate_source'] == 'none'
    assert 'candidates' not in obs.navigation
    env.close()
    env = make_matterport(monkeypatch, tmp_path, candidate_source='simulator')
    obs = env.reset()
    assert len(obs.navigation['candidates']) == 1
    assert env.env.settings['setRestrictedNavigation'] == (True,)
    assert env.env.actions == []
    assert set(obs.navigation['candidates'][0]) == {
        'action_id', 'viewpoint_id', 'relative_heading', 'relative_elevation', 'distance'}
    env.step(1)  # Still the original primitive FORWARD action.
    assert env.env.current == 'b'


def test_stable_candidate_ids_survive_reordering_and_validate_current_reachability(monkeypatch, tmp_path):
    env = make_matterport(monkeypatch, tmp_path, action_mode='candidate_viewpoint',
        ctrl_type='candidate_viewpoint', candidate_source='simulator', candidate_scope='all_neighbors')
    obs = env.reset()
    assert env.env.settings['setRestrictedNavigation'] == (False,)
    ids = {c['viewpoint_id']: c['action_id'] for c in obs.navigation['candidates']}
    assert len(ids) == 2 and env.env.actions == []
    with pytest.raises(ValueError, match='ctrl_type=candidate_viewpoint'):
        env.step(MetaAction(ctrl_space='navigation', ctrl_type='discrete', action=np.array([ids['c']])))
    env.env.reverse = True
    env.step(MetaAction(ctrl_space='navigation', ctrl_type='candidate_viewpoint', action=np.array([ids['c']])))
    assert env.env.current == 'c'
    assert env.env.actions[-1] == ([1], [0.], [0.])  # c changed from native index 2 to 1.
    assert env.travelled == 2.
    before = len(env.env.actions)
    with pytest.raises(ValueError, match='not currently navigable'):
        env.step(ids['b'])
    for invalid in [-1, .5, float('nan'), 999]:
        with pytest.raises(ValueError):
            env.step(invalid)
    assert len(env.env.actions) == before
    _, reward, done, info = env.step(0)
    assert done and info['success'] and reward == 1.


def candidate_observations():
    return [MetaObs(state=np.zeros(3, dtype=np.float32), raw_lang=instruction,
        navigation={'action_mode': 'candidate_viewpoint', 'candidate_source': 'simulator',
                    'candidates': [{'action_id': action} for action in actions]})
        for instruction, actions in [('Go left.', [2, 5]), ('Stop here.', [])]]


def test_ragged_candidates_survive_vectorization_and_meta_policy():
    from vlastudio.data_utils.normalize import Identity
    observations = candidate_observations()
    batched = organize_obs(observations)
    assert [len(n['candidates']) for n in batched.navigation] == [2, 0]
    class Policy:
        def select_action(self, obs):
            assert obs['raw_lang'] == ['Go left.', 'Stop here.']
            assert [len(n['candidates']) for n in obs['navigation']] == [2, 0]
            return torch.tensor([[[5.]], [[0.]]])
    normalizer = Identity(ctrl_space='navigation', ctrl_type='candidate_viewpoint')
    policy = MetaPolicy(Policy(), state_normalizer=normalizer, action_normalizer=normalizer,
                        ctrl_space='navigation', ctrl_type='candidate_viewpoint')
    actions = policy.inference(batched)[0]
    assert [int(a['action'][0]) for a in actions] == [5, 0]
    assert all(a['ctrl_type'] == 'candidate_viewpoint' for a in actions)
    assert organize_obs([MetaObs(state=np.zeros(3)), observations[1]]).navigation[0] is None


@pytest.mark.parametrize('transport', ['shm', 'fastapi'])
def test_server_batching_preserves_each_clients_candidates_and_instruction(transport):
    if transport == 'shm':
        from vlastudio.deploy.comm.shm.server import SHMPolicyServer as Server
    else:
        pytest.importorskip('fastapi')
        pytest.importorskip('uvicorn')
        from vlastudio.deploy.comm.fastapi.server import BatchedInferenceManager as Server
    observations = candidate_observations()
    for obs in observations:
        obs.to_batch()
    batched = Server._batch_metaobs(Server.__new__(Server), observations)
    assert batched.raw_lang == ['Go left.', 'Stop here.']
    assert [len(n['candidates']) for n in batched.navigation] == [2, 0]


def test_candidates_survive_actual_shared_memory_roundtrip():
    from vlastudio.deploy.shm_utils import SharedMemoryChannel
    obs = organize_obs(candidate_observations())
    name = 'vla_candidates_' + uuid.uuid4().hex
    writer = SharedMemoryChannel(name, max_size_mb=1, is_writer=True)
    reader = SharedMemoryChannel(name, is_writer=False)
    try:
        writer.write(asdict(obs))
        decoded = dict2meta(reader.read(copy=True))
        assert decoded.navigation == obs.navigation
        assert decoded.raw_lang == obs.raw_lang
    finally:
        reader.destroy()
        writer.destroy()


def test_candidates_survive_http_codec_and_legacy_prep():
    pytest.importorskip('fastapi')
    pytest.importorskip('uvicorn')
    from vlastudio.deploy.comm.fastapi.server import _to_jsonable, _payload_to_metaobs
    from vlastudio.deploy.comm.fastapi.codec import prepare_meta_obs_for_remote
    obs = organize_obs(candidate_observations())
    payload = json.loads(json.dumps(_to_jsonable(asdict(obs))))
    assert _payload_to_metaobs(payload).navigation == obs.navigation
    assert prepare_meta_obs_for_remote(obs).navigation == obs.navigation
