# Python library and worker internals

Install dependencies following the [environment guide](environments.md). The
default installation includes ACT, DP, shared data processing and ALOHA-Sim.
The [homepage](../README.md) shows workflows written as ordinary Python files.

`import vlastudio` uses only the standard library. Configurations, policies,
datasets, simulators and optional device SDKs are resolved lazily. The public
loaders return configuration handles; they do not immediately allocate models or
start a simulator. Their dependency checks inspect distribution metadata.

`vla.train`, `vla.serve`, `Environment.evaluate`, `vla.collect` and `vla.infer`
launch the existing workers using `sys.executable`. This preserves the original
policy-specific processors, collators, trainers and deployment code. Workers
stream logs and preserve the caller's working directory. They never create a
virtual environment, invoke pip/uv, or patch installed distributions.

Guard workflow entrypoints with `if __name__ == "__main__":`, especially for
simulator/device processes on Windows. Scripts are launched with `python run.py`.

## Paths and caches

Built-in YAML files live under `src/vlastudio/configs`. User YAML files can live
anywhere; relative data/checkpoint paths resolve from the current working directory.
Loaders accept built-in names and explicit YAML paths. Use `config_path` and
`plugin_path` options or `VLASTUDIO_CONFIG_PATH` and `VLASTUDIO_PLUGIN_PATH` for
additional lookup locations.

`VLASTUDIO_CACHE` selects the cache root; `model_cache_dir` and `data_cache_dir`
can separate weights from dataset caches. Cache configuration does not install
packages. The Python interpreter is always the one selected by the user.

## Extensions and dependency metadata

Policies supply `load_model` and a model with `select_action`. Optional processors,
collators and trainers customize training. Custom simulators implement the same
benchmark interface as the built-in examples. Extension references accept module
paths, `module:symbol` references and Python file references via `vla.resolve`.

Environment YAML files describe dependencies using `python`, `requirements`,
`extends`, optional platform restrictions and notes. A child can replace an
inherited package constraint; composing peer components retains their constraints
so conflicts are reported instead of silently discarded. Model-specific pinned
requirements files are committed for pip/uv to consume externally.

`runtime_manifest` selects dependency metadata and an optional custom worker
entrypoint. It does not select another interpreter or install anything.
`runtime="managed"`, environment creation and installation APIs have been removed.
Historical console task aliases dispatch into the current interpreter only;
new usage examples use library calls from user-authored Python files.

`check_dependencies=False` skips metadata checks for configuration orchestration.
Runtime imports can still fail if required packages are absent. Metadata checks
cannot validate CUDA, drivers, native libraries, assets, extras or exact VCS revisions.
