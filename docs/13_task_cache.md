# Task cache

ILStudio can persist either of two input-pipeline boundaries:

- `level: collator`: `dataset -> normalizer -> task transforms -> policy processor -> policy collator -> cache`;
- `level: processor`: `dataset -> normalizer -> task transforms -> policy processor -> cache -> policy collator`.

At collator level, training uses `CacheDataset` and the generic `CacheCollator`.
At processor level, `CacheBeforeCollatorDataset` restores individual processed
samples and the policy's original collator remains active. In both modes the
source dataset, normalizer, transforms, and processor are bypassed on cache hits.

## Task configuration

Caching is opt-in. The presence of the top-level `cache` section enables it;
when `cache` is absent (or set to `false`), the existing uncached path is used.
All cache settings live inside this section.

```yaml
cache:
  # Optional: collator (default) or processor. Processor level is safer for
  # policies whose collator depends on relationships within the training batch.
  level: collator

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

  # Cache construction always uses DataLoader(shuffle=False, batch_size=1).
  # num_workers defaults to the training config's dataloader_num_workers;
  # set it explicitly here to tune cache-build parallelism independently.
  num_workers: 8

  # Number of ready records queued by each build worker. Ignored when
  # num_workers is zero. Defaults to dataloader_prefetch_factor, then 2.
  prefetch_factor: 2

  # Optional field-aware image storage. Field paths refer to the saved payload:
  # processor output for level=processor, or collator output for level=collator.
  # This PI0 example stores float32 images in [-1, 1] as uint8 and restores them
  # before the next active pipeline stage.
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
   signatures of the stages included in the selected cache level.

Changing `cache.level` or any included input creates a new cache. At processor
level, the collator signature is excluded from the dataset hash. The default
policy namespace still contains the complete policy module source hash; set a
stable `cache.policy_name` when a collator-only policy source change should
intentionally reuse processor-level records. Writes are locked and atomic,
so concurrent distributed ranks wait for the first builder and then reuse the
completed cache.

Cache construction parallelizes source I/O, the policy processor, image storage
conversion, and, at collator level, the original policy collator in DataLoader
workers. Completed records are returned in source-index order to a single writer,
so HDF5 remains single-writer safe and NPY shards still close only at episode
boundaries. The worker count and prefetch depth are build-performance settings
and do not change cache identity. Set `cache.num_workers: 0` only for debugging
or for source datasets that already own an incompatible multiprocessing pipeline.

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
in the selected level's payload are quantized. Processor-level records are
decoded by `CacheBeforeCollatorDataset` before reaching the policy collator;
collator-level records are decoded by `CacheCollator`. With PI0's linear
uint8-to-`[-1, 1]` conversion, the configured uint8 round trip reconstructs the
original pixel values; other policies must declare their actual value range and
restoration dtype.

If processor output images are already uint8, leave `image_storage` disabled;
the original policy collator should receive those uint8 values unchanged.

HDF5 remains available for portability. It uses one file per configured dataset
and a contiguous payload stream, but can become a shared-file hotspot on
multi-node QZ jobs. Prefer `npy` for distributed training.

Cache files contain trusted Python-serialized model inputs. Do not load caches
from untrusted users or locations.
