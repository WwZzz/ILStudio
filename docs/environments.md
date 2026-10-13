# Environments

Explicit `env` commands install dependencies. Imports and workflow API calls never
modify environments.

```bash
pip install vlastudio
vlastudio env install                         # current interpreter, registered as base
vlastudio env create                          # isolated base: ACT, DP, ALOHA-Sim
vlastudio env create -n pi05 --policy openpi
vlastudio env create -n smol --policy smolvla --env libero
vlastudio env run pi05 -- python train.py
vlastudio env list
vlastudio env check pi05
vlastudio env check pi05 --smoke
```

Add dependencies to a named environment using its interpreter:

```bash
vlastudio env run smol -- vlastudio env install -n smol --env metaworld
```

`env activate NAME` opens a child shell; `exit` returns to the previous shell.
`env deactivate` opens a base child shell. No shell initialization or `eval` is
required. A subprocess cannot modify its parent shell. Omitted names default to
base, except `create --policy/--env`, which derives a name from the component
(ACT, MLP and DP use base).

## Native simulators

Review the upstream licenses before using acceptance flags:

```bash
vlastudio env create -n rlbench --env rlbench --accept-license coppeliasim-edu
vlastudio env create -n behavior --env behavior --accept-license nvidia-isaac-sim
vlastudio env create -n robotwin --env robotwin --assets
vlastudio env run rlbench -- xvfb-run -a python evaluate.py
```

RLBench installs pinned PyRep/RLBench and CoppeliaSim EDU 4.1.0. The host needs a
C compiler and OpenGL/Qt libraries; headless hosts also need Xvfb and xauth.
[CoppeliaSim licensing](https://www.coppeliarobotics.com/licensing).

BEHAVIOR installs OmniGibson 3.7.2 and Isaac Sim 4.5.0. It requires Linux, Python
3.10, glibc >= 2.34 and a compatible NVIDIA graphics driver. Add `--assets` and
`--accept-license behavior-dataset` to download datasets (a large download).
[Isaac Sim license](https://docs.omniverse.nvidia.com/isaacsim/latest/common/licenses.html),
[BEHAVIOR dataset license](https://behavior.stanford.edu/behavior-1k-dataset-license.html).

Host drivers are not installed automatically. On an apt-based host, explicitly
add `--system-packages` as root to install the required OS libraries. Otherwise
missing prerequisites are reported; wheel platform tags are never changed.

RoboTwin installs pinned source, SAPIEN, mplib, PyTorch3D and cuRobo. It requires
a CUDA toolkit compatible with PyTorch 2.4.1; set `CUDA_HOME` if needed. `--assets`
downloads the official textures, embodiments and objects, then configures their
paths. The CUDA extensions are compiled in the selected environment.

## Storage and validation

`VLASTUDIO_CACHE` (default `~/.cache/vlastudio`) contains `envs/NAME`, shared
`runtimes/`, `downloads/`, `assets/`, `registry.json`, and `installations/NAME/`
logs, resolved package lists and check records. Registered current environments
stay in their original locations. A moved interpreter is reported as missing.

The registry distinguishes registration, installation, import checks and smoke
checks. Smoke checks exercise simulator reset/step or a CUDA forward/backward;
they do not measure policy quality or benchmark success rates. Model weights and
training datasets are separate from environment installation.

Validated on Linux with an RTX 4090 (2026-10-13):

| Recipe | Verification |
| --- | --- |
| Base (ACT / DP / ALOHA-Sim) | Policy imports, CUDA forward/backward, simulator render/reset/step |
| SmolVLA, OpenPI | Policy imports and CUDA forward/backward |
| LIBERO, MetaWorld | Simulator render/reset/step |
| RLBench | ReachTarget render/reset/step |
| RoboTwin | adjust_bottle render/reset/step with both cuRobo planners; PyTorch3D and Warp GPU kernels |
| BEHAVIOR | Installation/imports passed; simulator verification pending (native Isaac Sim also crashes on the test host) |

RoboTwin's task check uses `demo_clean`; it does not validate every task or
randomized-background configuration. Large resource downloads resume completed
parts and verify the upstream SHA-256 before extraction. Set
`VLASTUDIO_DOWNLOAD_WORKERS` (1–64, default 8) to adjust download concurrency.

Use `-i URL` on create/install for a custom index. No mirror is selected by
default. `--dry-run` prints a recipe without installing it. Retry a failed install
with the same command; existing environments are never deleted.

## Custom components

A component config may point to a runtime YAML, or use
`--runtime-manifest /path/runtime.yaml`:

```yaml
python: '3.11'
requirements:
  - torch==2.7.1
  - my-policy-package==1.0.0
```

Recipes support `extends`, `overrides`, `overlays`, platform constraints and named
native `installers`. Peer constraints are kept together so conflicts are reported.
Use separate model and simulator environments with remote inference when needed.

`pip install "vlastudio[base]"` is available for users managing their own venvs.
OpenPI needs pinned source dependencies, overrides and a Transformers patch;
`env create --policy openpi` performs these steps explicitly during installation.
