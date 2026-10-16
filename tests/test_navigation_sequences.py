"""Dataset temporal sampling and separation from model-specific encodings."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlastudio.data_utils.datasets.navigation import _NavigationDataset


class ReplayFixture(_NavigationDataset):
    def __init__(self, root, mode='episode', chunk_size=1, offsets=None):
        self.episodes = [{'episode_id': 'a'}, {'episode_id': 'b'}]
        self.episode_lengths = [3, 2]
        self._sampling(mode, chunk_size, offsets)
        self._cache({'test': 'temporal'}, root)
        self.replayed = 0

    def _replay(self, episode):
        self.replayed += 1
        length = 3 if episode['episode_id'] == 'a' else 2
        for i in range(length):
            yield dict(image=np.full((1, 3, 4, 5), i, np.uint8),
                       depth=np.full((1, 4, 5), i/10, np.float32),
                       state=np.array([i, 0, 1], np.float32), raw_lang='Go to the door.',
                       action=torch.tensor([[i+1 if i+1 < length else 0]]),
                       navigation={'viewpoint_id': f'{episode["episode_id"]}-{i}'},
                       supervision={'distance_to_goal': float(length-i), 'initial_distance_to_goal': float(length)})


def test_dataset_owns_whole_episode_sampling_and_raw_cache(tmp_path):
    dataset = ReplayFixture(tmp_path)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample['image'].shape == (1, 3, 3, 4, 5)
    assert sample['depth'].shape == (1, 3, 4, 5)
    assert sample['state'].shape == (3, 3)
    assert sample['action'].tolist() == [[1], [2], [0]]
    assert sample['image'].dtype == torch.uint8
    assert not sample['is_pad'].any()
    assert sample['frame_ids'].tolist() == sample['action_frame_ids'].tolist() == [0, 1, 2]
    assert sample['navigation'][2]['viewpoint_id'] == 'a-2'
    assert {'instruction', 'rgb_features', 'candidate_features', 'labels'}.isdisjoint(sample)
    assert not hasattr(dataset, 'model')
    dataset[0]
    assert dataset.replayed == 1
    assert dataset[1]['episode_id'] == 'b'


def test_window_sampling_clamps_without_crossing_episodes(tmp_path):
    dataset = ReplayFixture(tmp_path, 'window', 4)
    sample = dataset[2]
    assert len(dataset) == 5
    assert sample['frame_ids'].tolist() == [2, 2, 2, 2]
    assert sample['is_pad'].tolist() == [False, True, True, True]
    assert torch.equal(sample['is_pad'], sample['image_is_pad'])
    assert [row['viewpoint_id'] for row in sample['navigation']] == ['a-2']*4
    assert sample['action'].tolist() == [[0]]*4
    assert dataset[3]['episode_id'] == 'b'


def test_fastwam_style_image_offsets_and_single_frame_shape(tmp_path):
    sample = ReplayFixture(tmp_path, 'window', 3, [0, 2])[1]
    assert sample['image'].shape == (1, 2, 3, 4, 5)
    assert sample['frame_ids'].tolist() == [1, 2]
    assert sample['image_is_pad'].tolist() == [False, True]
    assert sample['is_pad'].tolist() == [False, False, True]
    single = ReplayFixture(tmp_path, 'window')[0]
    assert single['image'].shape == (1, 3, 4, 5)
    assert single['depth'].shape == (1, 4, 5)
    assert single['state'].shape == (1, 3)


@pytest.mark.parametrize('mode,size,offsets', [('invalid', 1, None), ('window', 0, None),
    ('window', 2, [1]), ('window', 2, [0, 0]), ('episode', 1, [0])])
def test_invalid_temporal_configuration(tmp_path, mode, size, offsets):
    with pytest.raises(ValueError):
        ReplayFixture(tmp_path, mode, size, offsets)


def test_model_processor_trims_padding_without_resampling(tmp_path):
    from vlastudio.policy.navigation_processing import NavigationProcessor

    class Model:
        def candidate_inputs(self, navigation, states, panoramas):
            assert len(navigation) == 2
            assert states.shape == (2, 3)
            return torch.zeros(2, 3, 2176), torch.zeros(2, 128), torch.ones(2, 3, dtype=torch.bool), [[17, 1, 0], [17, 0]]

    sample = ReplayFixture(tmp_path, 'window', 4)[1]
    sample['visual_features'] = torch.zeros(4, 36, 2048)
    sample['action'][:2] = torch.tensor([[1], [0]])
    processor = NavigationProcessor(Model(), 'vlnbert')
    encoded = processor(sample)
    assert encoded['labels'].tolist() == [1, 1]
    assert encoded['action'].shape == (2, 1)
    assert not hasattr(processor, 'adapt_dataset')
    assert not hasattr(processor, 'feature_cache')


def test_tokenizer_preserves_annotation_hyphens_and_punctuation():
    pytest.importorskip('gym')
    pytest.importorskip('yacs')
    from vlastudio.policy.vlnce_cma.tokenizer import InstructionTokenizer
    vocab = ['<pad>', '<unk>', 'walk', 'cream-colored', 'room', '.', 'owner', "'", 's']
    tokenize = InstructionTokenizer(vocab, download=False)
    assert tokenize('Walk, cream-colored room?', length=6) == [2, 3, 4, 0, 0, 0]
    assert tokenize("Owner's room.", length=6) == [6, 7, 8, 4, 5, 0]


def test_panorama_data_reader_is_lazy_and_picklable(tmp_path, monkeypatch):
    import json
    import pickle
    from vlastudio.data_utils.datasets.navigation_features import PanoramaFeatureStore
    from vlastudio.utils import navigation_assets
    np.save(tmp_path/'features.npy', np.arange(36*2048, dtype=np.float32).reshape(1, 36, 2048))
    (tmp_path/'index.json').write_text(json.dumps({'scan_current': 0}))
    calls = []
    def resolve(path, download):
        calls.append((path, download))
        return tmp_path
    monkeypatch.setattr(navigation_assets, 'panorama_features', resolve)
    store = PanoramaFeatureStore(tmp_path, download=False)
    assert not calls
    value = store.read('scan', 'current')
    assert value.shape == (36, 2048)
    value[:] = 0
    assert store.read('scan', 'current')[1, 0] == 2048
    restored = pickle.loads(pickle.dumps(store))
    assert restored._features is None
    np.testing.assert_array_equal(restored.read('scan', 'current'), store.read('scan', 'current'))


def test_cma_processor_encodes_without_disk_cache_and_tracks_encoder_weights(tmp_path):
    pytest.importorskip('gym')
    pytest.importorskip('yacs')
    from vlastudio.policy.navigation_processing import NavigationProcessor
    from vlastudio.policy.vlnce_cma.tokenizer import InstructionTokenizer

    class Depth(torch.nn.Conv2d):
        def forward(self, observation):
            return super().forward(observation['depth'].permute(0, 3, 1, 2))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = torch.nn.Conv2d(3, 4, 1)
            self.depth = Depth(1, 2, 1)
            self.net = SimpleNamespace(rgb_encoder=SimpleNamespace(cnn=self.rgb),
                                       depth_encoder=SimpleNamespace(visual_encoder=self.depth))
            self.instruction_tokenizer = InstructionTokenizer(['<pad>', '<unk>'])
            self.requires_grad_(False)

    sample = ReplayFixture(tmp_path)[0]
    before = set(tmp_path.rglob('*'))
    model = Model()
    processor = NavigationProcessor(model, 'cma')
    row = processor(sample)
    assert row['rgb_features'].shape == (3, 4, 4, 5)
    assert row['depth_features'].shape == (3, 2, 4, 5)
    torch.testing.assert_close(row['progress'], torch.tensor([0., 1/3, 2/3]))
    assert set(tmp_path.rglob('*')) == before
    with torch.no_grad():
        model.rgb.weight.add_(1)
    assert NavigationProcessor(model, 'cma').encoding_signature != processor.encoding_signature
    model.rgb.requires_grad_(True)
    assert 'rgb' in NavigationProcessor(model, 'cma')(sample)


def test_empty_navigation_state_survives_shared_memory_transport():
    from dataclasses import asdict
    import uuid
    from vlastudio.benchmark.base import MetaObs, dict2meta
    from vlastudio.benchmark.utils import organize_obs
    from vlastudio.deploy.shm_utils import SharedMemoryChannel
    observation = organize_obs([MetaObs(state=np.empty(0, np.float32), raw_lang='Go left.') for _ in range(2)])
    name = 'vla_empty_state_'+uuid.uuid4().hex
    writer = SharedMemoryChannel(name, max_size_mb=1, is_writer=True)
    reader = SharedMemoryChannel(name, is_writer=False)
    try:
        writer.write(asdict(observation))
        restored = dict2meta(reader.read(copy=True))
        assert restored.state.shape == (2, 0)
        assert restored.raw_lang == ['Go left.']*2
    finally:
        reader.destroy()
        writer.destroy()
