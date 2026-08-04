# Task cache

ILStudio can persist the output of the complete input pipeline:

`dataset -> normalizer -> task transforms -> policy processor -> policy collator`

On a cache hit, training replaces that pipeline with `CacheDataset` and
`CacheCollator`. The source dataset is not opened by training workers.

## Task configuration

Caching is opt-in. The presence of the top-level `cache` section enables it;
when `cache` is absent (or set to `false`), the existing uncached path is used.
All cache settings live inside this section.

```yaml
cache:
  # Optional. Defaults to $HF_HOME/ilstd_cache. If HF_HOME is unset, the
  # fallback is ~/.cache/huggingface/ilstd_cache.
  root: /inspire/hdd/global_user/<user>/ilstd_cache

  # Optional. By default the namespace contains the policy module path and
  # source hash. Use this only when policies intentionally share a pipeline.
  policy_name: pi-family-v1

  # Optional backend and episode-aligned target shard size.
  format: npy
  shard_size_mb: 512

  # Bound the process-wide mmap pool in every DataLoader worker. Offset arrays
  # are kept in RAM and do not consume a persistent file descriptor.
  max_open_shards: 16

  # Optional node-local staging. It can also be supplied by ILSTD_CACHE_LOCAL.
  local_dir: /tmp/ilstd_cache

  # Optional field-aware image storage. This PI0 example converts the collator's
  # float32 BCHW images in [-1, 1] to uint8 on disk and restores float32 values
  # before returning the training batch.
  image_storage:
    dtype: uint8
    fields:
      - observation.image.*
    value_range: [-1.0, 1.0]
    restore_dtype: float32
    strict_range: true
```

An empty mapping (`cache: {}`) enables caching with defaults. The former
top-level `enable_cache` and `cache_*` keys are rejected to avoid two competing
configuration layouts.

The persistent path has two identity components:

1. a policy namespace derived from the policy module source hash, or the
   explicit `cache.policy_name`;
2. a dataset hash derived from the normalized dataset configuration, task
   normalization/transforms, cache format/image storage, and
   processor/collator signature.

Changing any of those inputs creates a new cache. Writes are locked and atomic,
so concurrent distributed ranks wait for the first builder and then reuse the
completed cache.

## Build explicitly

`scripts/create_task_cache.py` accepts the same policy, task, training config, output,
evaluation ratio, and dotted override arguments as `train.py`:

```bash
python scripts/create_task_cache.py \
  --policy pi0_lora \
  --task /path/to/task.yaml \
  --training_config default
```

`train.py` performs the same build automatically when the task contains a
`cache` section and one or more required cache entries are missing.

## Format and distributed-I/O behavior

The default NumPy backend writes episode-aligned shard pairs:

- `payload-XXXXX.npy`: a contiguous uint8 record stream, opened with NumPy
  memory mapping;
- `offsets-XXXXX.npy`: the per-sample byte offsets, loaded into RAM so they do
  not hold another file descriptor;
- `manifest.json`: dataset identity, episode ranges, shard ranges, and collation
  metadata.

It deliberately does not use `.npz`: ZIP members cannot provide useful random
memory mapping and compressed members add CPU decompression to the training hot
path.

In distributed mode, when weighted sampling is not configured, ILStudio assigns
contiguous cache-shard ranges to ranks instead of striding every rank across the
whole dataset. This reduces shared filesystem fan-out and single-file contention.
When `cache.local_dir` is set, each rank/worker stages only the shards it touches,
guarded by a node-local lock. Tasks that set `sample_weights` retain weighted
sampling semantics and therefore may read across more shards.

The payload mmap pool is process-wide and LRU-bounded by
`cache.max_open_shards`. This matters when many small configured datasets each
produce a shard: the number of persistent mmap handles per worker remains
bounded instead of growing for the lifetime of the worker.

`cache.image_storage` is optional and never guessed. Only configured field paths
are quantized. With PI0's linear uint8-to-`[-1, 1]` conversion, the configured
uint8 round trip reconstructs the original collated pixel values; other policies
must declare their actual value range and restoration dtype.

HDF5 remains available for portability. It uses one file per configured dataset
and a contiguous payload stream, but can become a shared-file hotspot on
multi-node QZ jobs. Prefer `npy` for distributed training.

Cache files contain trusted Python-serialized model inputs. Do not load caches
from untrusted users or locations.
