# Inspire project context

## Path conventions

- Repository: `/inspire/hdd/global_user/wangzheng-240308120196/ILStudio`
- Durable task caches should stay under an `/inspire/...` shared path or the
  user's shared `HF_HOME`.
- Optional `cache_local_dir` content is ephemeral node-local staging and must
  not be treated as the durable cache.

## Existing notebook

- `minimalrl-4090-0804` is the current interactive development notebook used
  for this repository.

## Workload guidance

- Use the interactive notebook for cache smoke tests and short probes.
- Use a QZ distributed-training Job for multi-node throughput validation.
