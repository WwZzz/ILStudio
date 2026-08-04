from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from data_utils.task_cache import (
    CacheCollator,
    CacheDataset,
    CacheDescriptor,
    TaskCacheManager,
    _build_cache,
    hash_policy_module,
)
from configs.loader import ConfigLoader
from data_utils import data_loader


class ToyDataset:
    def __init__(self):
        self.samples = [
            {"episode_id": 0, "length": 2, "state": [1.0, 2.0]},
            {"episode_id": 0, "length": 4, "state": [3.0, 4.0]},
            {"episode_id": 1, "length": 3, "state": [5.0, 6.0]},
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(self.samples[index])


class ToyProcessor:
    def __call__(self, sample):
        ids = torch.arange(1, sample["length"] + 1, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "state": torch.tensor(sample["state"], dtype=torch.float32),
        }


class ToyCollator:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="left", pad_token_id=9)

    def __call__(self, instances):
        max_length = max(instance["input_ids"].shape[0] for instance in instances)
        input_ids = []
        labels = []
        attention_mask = []
        for instance in instances:
            pad = max_length - instance["input_ids"].shape[0]
            input_ids.append(
                torch.cat((torch.full((pad,), 9), instance["input_ids"]))
            )
            labels.append(
                torch.cat((torch.full((pad,), -100), instance["labels"]))
            )
            attention_mask.append(
                torch.cat((torch.zeros(pad, dtype=torch.bool), torch.ones(instance["input_ids"].shape[0], dtype=torch.bool)))
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "state": torch.stack([instance["state"] for instance in instances]),
        }


def make_descriptor(root: Path, cache_format: str) -> CacheDescriptor:
    cache_dir = root / cache_format
    return CacheDescriptor(
        index=0,
        kind="dataset",
        name="toy",
        config={"name": "toy"},
        config_hash="config-hash",
        dataset_hash=f"dataset-hash-{cache_format}",
        cache_dir=cache_dir,
        cache_file=cache_dir / "cache.hdf5",
        manifest_file=cache_dir / "manifest.json",
    )


def build_toy_cache(tmp_path: Path, cache_format: str) -> tuple[CacheDataset, ToyProcessor, ToyCollator]:
    descriptor = make_descriptor(tmp_path, cache_format)
    processor = ToyProcessor()
    collator = ToyCollator()
    common_manifest = {
        "policy": {"cache_key": "toy-policy"},
        "collation_spec": {
            "padding_side": "left",
            "pad_token_id": 9,
            "label_pad_token_id": -100,
        },
    }
    _build_cache(
        dataset=ToyDataset(),
        processor=processor,
        collator=collator,
        descriptor=descriptor,
        common_manifest=common_manifest,
        cache_format=cache_format,
        shard_size_bytes=1,
    )
    return CacheDataset(descriptor.cache_file, descriptor.manifest_file), processor, collator


def assert_batch_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def test_npy_cache_round_trip_and_episode_sharding(tmp_path):
    dataset, processor, original_collator = build_toy_cache(tmp_path, "npy")
    assert len(dataset) == 3
    assert len(dataset.manifest["shards"]) == 2

    actual = CacheCollator(dataset.collation_spec)([dataset[0], dataset[2]])
    source = ToyDataset()
    expected = original_collator([processor(source[0]), processor(source[2])])
    assert_batch_equal(actual, expected)


def test_hdf5_cache_round_trip(tmp_path):
    dataset, processor, original_collator = build_toy_cache(tmp_path, "hdf5")
    actual = CacheCollator(dataset.collation_spec)([dataset[1], dataset[2]])
    source = ToyDataset()
    expected = original_collator([processor(source[1]), processor(source[2])])
    assert_batch_equal(actual, expected)


def test_npy_cache_can_stage_shards_locally(tmp_path):
    dataset, _, _ = build_toy_cache(tmp_path, "npy")
    staged = CacheDataset(
        dataset.cache_file,
        dataset.manifest_file,
        local_cache_dir=tmp_path / "local",
    )
    staged[0]
    staged_files = list((tmp_path / "local").rglob("*.npy"))
    assert len(staged_files) == 2


def test_npy_cache_works_with_dataloader_workers(tmp_path):
    dataset, _, _ = build_toy_cache(tmp_path, "npy")
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=CacheCollator(dataset.collation_spec),
    )

    assert next(iter(loader))["input_ids"].shape[0] == 2


def test_real_pi0_task_config_resolves_cache_overrides():
    loader = ConfigLoader(
        unknown_args=[
            "--task.enable_cache",
            "true",
            "--task.cache_format",
            "npy",
        ]
    )
    task_config, _ = loader.load_task("sim_transfer_cube_scripted")
    policy_config, _ = loader.load_policy("pi0_lora")
    module_hash = hash_policy_module(policy_config["module_path"])

    assert task_config["enable_cache"] is True
    assert task_config["cache_format"] == "npy"
    assert policy_config["module_path"] == "policy.openpi"
    assert len(module_hash) == 64


def test_distributed_sampler_keeps_ranks_on_disjoint_shards(monkeypatch):
    dataset = SimpleNamespace(
        shard_ranges=[(0, 4), (4, 8), (8, 12)],
        __len__=lambda: 12,
    )
    monkeypatch.setattr(data_loader.dist, "get_world_size", lambda: 2)

    rank_indices = []
    for rank in range(2):
        monkeypatch.setattr(data_loader.dist, "get_rank", lambda rank=rank: rank)
        sampler = data_loader.CacheShardDistributedSampler(
            dataset, shuffle=False, seed=0
        )
        rank_indices.append(list(sampler))

    assert len(rank_indices[0]) == len(rank_indices[1])
    assert set(rank_indices[0]).isdisjoint(set(rank_indices[1]))
    assert set(rank_indices[0]) | set(rank_indices[1]) == set(range(12))


def test_manager_builds_on_miss_and_reuses_on_hit(tmp_path, monkeypatch):
    calls = []

    def load_datasets_stub(args, task_config, save_norm=True):
        calls.append((args.output_dir, save_norm))
        return [ToyDataset()]

    monkeypatch.setattr("data_utils.utils.load_datasets", load_datasets_stub)
    args = SimpleNamespace(task="toy", output_dir=str(tmp_path / "output"), eval_ratio=0.0)
    task_config = {
        "enable_cache": True,
        "cache_root": str(tmp_path / "cache"),
        "cache_format": "npy",
        "cache_shard_size_mb": 1,
        "datasets": [
            {"type": "tests.ToyDataset", "name": "toy", "args": {}}
        ],
    }
    policy_config = {"module_path": "data_utils.task_cache"}

    first = TaskCacheManager(
        args, task_config, policy_config, ToyProcessor(), ToyCollator()
    )
    assert len(first.ensure()) == 1
    second = TaskCacheManager(
        args, task_config, policy_config, ToyProcessor(), ToyCollator()
    )
    assert len(second.ensure()) == 1
    assert len(calls) == 1


def test_cache_alias_still_hashes_effective_normalization(tmp_path):
    task_config = {
        "enable_cache": True,
        "cache_root": str(tmp_path / "cache"),
        "cache_policy_name": "shared-pipeline",
        "datasets": [
            {"type": "tests.ToyDataset", "name": "toy", "args": {}}
        ],
    }
    policy_config = {"module_path": "data_utils.task_cache"}
    first_args = SimpleNamespace(task="toy", action_normalize="minmax")
    second_args = SimpleNamespace(task="toy", action_normalize="zscore")

    first = TaskCacheManager(
        first_args, task_config, policy_config, ToyProcessor(), ToyCollator()
    )
    second = TaskCacheManager(
        second_args, task_config, policy_config, ToyProcessor(), ToyCollator()
    )

    assert first.policy_key == second.policy_key == "shared-pipeline"
    assert first.descriptors[0].dataset_hash != second.descriptors[0].dataset_hash


def test_cached_dataset_get_dataloader_returns_loader_not_nested_tuple(tmp_path):
    dataset, _, _ = build_toy_cache(tmp_path, "npy")
    args = SimpleNamespace(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        background_prefetch=False,
        seed=0,
    )
    train_loader, eval_loader = data_loader.get_dataloader(
        dataset,
        None,
        processor=None,
        collator=CacheCollator(dataset.collation_spec),
        args=args,
    )

    assert isinstance(train_loader, DataLoader)
    assert eval_loader is None
    assert next(iter(train_loader))["input_ids"].shape[0] == 2
