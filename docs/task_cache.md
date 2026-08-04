# Task cache

ILStudio can persist the output of the complete input pipeline:

`dataset -> normalizer -> task transforms -> policy processor -> policy collator`

On a cache hit, training replaces that pipeline with `CacheDataset` and
`CacheCollator`. The source dataset is not opened by training workers.

## Task configuration

Caching is opt-in and disabled when `enable_cache` is absent.

```yaml
enable_cache: true

# Optional. Defaults to $HF_HOME/ilstd_cache. If HF_HOME is unset, the fallback
# is ~/.cache/huggingface/ilstd_cache.
cache_root: /inspire/hdd/global_user/<user>/ilstd_cache

# Optional. By default the namespace contains the policy module path and source
# hash. Set an explicit name only when policies intentionally share an identical
# input pipeline.
cache_policy_name: pi-family-v1

# Optional. "npy" is the default; "hdf5" is the compatibility backend.
cache_format: npy

# Optional. Target shard size. A shard closes at an episode boundary, so an
# individual large episode can exceed this value.
cache_shard_size_mb: 512

# Optional node-local staging directory. Keep the durable cache in cache_root;
# use this only when the QZ job provides enough ephemeral local SSD/RAM disk.
# It can also be supplied with ILSTD_CACHE_LOCAL.
cache_local_dir: /tmp/ilstd_cache
```

The persistent path has two identity components:

1. a policy namespace derived from the policy module source hash, or the
   explicit `cache_policy_name`;
2. a dataset hash derived from the normalized dataset configuration, task
   normalization/transforms, cache format, and processor/collator signature.

Changing any of those inputs creates a new cache. Writes are locked and atomic,
so concurrent distributed ranks wait for the first builder and then reuse the
completed cache.

## Build explicitly

`create_task_cache.py` accepts the same policy, task, training config, output,
evaluation ratio, and dotted override arguments as `train.py`:

```bash
python create_task_cache.py \
  --policy pi0_lora \
  --task /path/to/task.yaml \
  --training_config default
```

`train.py` performs the same build automatically when `enable_cache: true` and
one or more required cache entries are missing.

## Format and distributed-I/O behavior

The default NumPy backend writes episode-aligned shard pairs:

- `payload-XXXXX.npy`: a contiguous uint8 record stream, opened with NumPy
  memory mapping;
- `offsets-XXXXX.npy`: the per-sample byte offsets, also memory mapped;
- `manifest.json`: dataset identity, episode ranges, shard ranges, and collation
  metadata.

It deliberately does not use `.npz`: ZIP members cannot provide useful random
memory mapping and compressed members add CPU decompression to the training hot
path.

In distributed mode, when weighted sampling is not configured, ILStudio assigns
contiguous cache-shard ranges to ranks instead of striding every rank across the
whole dataset. This reduces shared filesystem fan-out and single-file contention.
When `cache_local_dir` is set, each rank/worker stages only the shards it touches,
guarded by a node-local lock. Tasks that set `sample_weights` retain weighted
sampling semantics and therefore may read across more shards.

HDF5 remains available for portability. It uses one file per configured dataset
and a contiguous payload stream, but can become a shared-file hotspot on
multi-node QZ jobs. Prefer `npy` for distributed training.

Cache files contain trusted Python-serialized model inputs. Do not load caches
from untrusted users or locations.
