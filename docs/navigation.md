# Navigation simulators

Navigation adapters share VLAStudio's `MetaObs` / `MetaAction`, config resolution,
environment factories, policy loading and `eval_sim` interfaces. Native simulators
are optional: importing VLAStudio does not import or install them.

| Config name | Backend | Task | Runtime status |
| --- | --- | --- | --- |
| `habitat.habitat_pointnav` | Habitat-Sim 0.3.3 | Continuous GPS/compass PointNav | Real training and evaluation in Example 09 |
| `habitat.habitat_vlnce` | Habitat-Lab 0.3.3 | Continuous R2R with RGBD and language | Real test-scene API fixture verified; benchmark needs MP3D assets |
| `matterport3d.matterport_r2r` | Matterport3DSimulator | Classic discrete-viewpoint R2R | Adapter; requires native build and authorized panoramas |

The continuous R2R config uses Habitat-Lab's built-in `VLN-v0` / `R2RVLN-v1` task.
It is a modern task port, not a reproduction of the original VLN-CE 0.1.7
leaderboard. PointNav acceptance does not establish instruction following or
generalization to unseen scenes. Matterport3D and continuous R2R have contract
tests but must be validated with real licensed assets before claiming benchmark
support is fully runtime-verified. `verify_habitat.py --lab` checks the native
PointNav renderer and Habitat-Lab R2R API with a clearly marked synthetic episode
on a public test scene; that fixture is not a benchmark score.

## Install Habitat-Sim on Python 3.10

Use Linux, Python 3.10 and a separate environment. Install VLAStudio's base
dependencies first. Habitat's public v0.3.3 Conda builds use Python 3.9; VLAStudio
requires Python >=3.10, so build the native extension for the current interpreter.

```bash
python -m venv .venv-navigation
source .venv-navigation/bin/activate
python -m pip install -e . pytest 'cmake>=3.24,<3.28' ninja 'setuptools<70' wheel
git clone --branch v0.3.3 --depth 1 https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
cd third_party/habitat-sim
python -m pip install -r requirements.txt
python setup.py build_ext --parallel 4 install --headless --no-bullet --no-lto --build-type Release
cd ../..
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data
```

If Git LFS is unavailable, download the selected official test scene directly:

```bash
python examples/09_navigation/download_test_scene.py
```

The build needs EGL/OpenGL and JPEG development headers, plus working NVIDIA EGL
drivers when RGBD rendering is enabled. If headers are unavailable on a shared
host, install them in a task-owned prefix, without changing system packages:

```bash
conda create -y -p "$PWD/.native-navigation" --override-channels \
  -c https://conda.anaconda.org/conda-forge libegl-devel libgl-devel libjpeg-turbo
export CMAKE_PREFIX_PATH="$PWD/.native-navigation"
export CPLUS_INCLUDE_PATH="$PWD/.native-navigation/include"
export LIBRARY_PATH="$PWD/.native-navigation/lib"
```

Run those commands before the source build and keep the prefix available.
Use `--no-render` in Example 09 for a navmesh-only run when no EGL renderer is
available; this remains real Habitat collision/navigation simulation, but does
not verify image rendering. Renderer status is saved in the result configuration.

## Train and accept a simple policy

```bash
python examples/09_navigation/train_and_evaluate.py \
  --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb \
  --output results/navigation
```

This trains a 3-input, two-layer 64-unit MLP with behavior cloning and DAgger.
Inputs are normalized goal distance, sine of bearing, and cosine of bearing.
The GPS/compass sensor is allowed in PointNav; the model does not consume language
or images. Expert actions are queried only during training. The checkpoint is
reloaded before evaluation.

Protocol: 200 training, 40 validation and 100 test episodes in one official
Habitat test scene; fixed disjoint seeds and unique start/goal/yaw tuples; sampled
geodesic distances 1–5m; 0.25m forward steps; 30-degree turns; at most 80 actions;
success requires explicit STOP at geodesic distance <0.2m. Checkpoint selection
uses validation SR then SPL. Test is evaluated once after selection. Uniform
random and immediate-STOP baselines run on the identical test episodes.

Acceptance requires test SR >=30% and SR above random. A failing run exits
nonzero. `evaluation.json` records all actions, episode metrics, training history,
arguments, versions and checkpoint SHA256. `episodes.json` fixes all splits.
`policy.pt` and its metadata can be reloaded by `policy.navigation_mlp`.
`rollout.mp4` records the first test episode when rendering is enabled.

The initial verified run achieved **87/100 test successes (87%), SPL 0.862260**;
random and immediate STOP each achieved 0/100. Training produced 13,260 labels
over three rounds, with zero expert failures. The selected checkpoint came from
round 1 (validation SR 90%). See [recorded acceptance](evidence/navigation/acceptance.json).

The policy implements the framework `load_model`, `select_action` and batched
`inference` contracts. Metadata uses identity normalization and
`ctrl_space: navigation`, `ctrl_type: discrete`; continuous action smoothing
or interpolation must not be used on discrete navigation actions.

Run the checkpoint through the standard VLAStudio inference worker, shared memory
and action manager on the same deterministic test seeds:

```bash
python -m vlastudio.entrypoints.eval_sim -m results/navigation \
  -e habitat.habitat_pointnav --device cpu -bs 0 -n 100 \
  -o results/navigation-framework \
  --env.scene /absolute/path/to/skokloster-castle.glb \
  --env.seed 200000 --env.max_timesteps 80 --env.reuse_env true
```

The navigation evaluator is selected by the environment factory's module. SPL
is averaged across rollouts rather than added.

## QZ mounted HM3D smoke run

QZ Data Square's active HM3D record (1887, `hm3d:v1`) was successfully mounted
at `/inspire/dataset/hm3d/v1`. It contains Habitat geometry tar archives; it is
not the original MP3D panorama/R2R release. Attach the live-validated
`dataset_info` entry to a new QZ instance instead of copying the whole dataset:

```json
{"dataset_info": [{"dataset_id": "hm3d", "version_id": "v1"}]}
```

For a small reproducible check, unpack only the first two scenes from the mounted
minival archive (59,974,964 bytes, including their navigation meshes):

```bash
python examples/09_navigation/prepare_mounted_hm3d.py \
  --archive /inspire/dataset/hm3d/v1/hm3d-minival-habitat-v0.2.tar \
  --output data/hm3d-smoke
python examples/09_navigation/train_and_evaluate.py \
  --scene data/hm3d-smoke/00800-TEEsavR23oF/TEEsavR23oF.basis.glb \
  --output results/navigation-hm3d
```

Selection uses scene names sorted before training. The first scene supplies
200/40/100 disjoint training/validation/test episodes with the protocol above.
The second scene, `00801-HaxA7YrQdEC`, supplies 100 additional episodes with seed
300000 and is never used for training or checkpoint selection.

| Evaluation | Success | SPL | Random / immediate STOP |
| --- | --- | --- | --- |
| First scene, held-out episodes | 75/100 | 0.737506 | 0/100 each |
| Second scene, unseen during training | 79/100 | 0.778237 | 0/100 each |

All three training rounds are retained in the evidence. Validation selected
round 0 (75%); later rounds reached only 25%. The final checkpoint passed the
30% threshold without selecting against test results. There were 17,025 expert
labels and zero expert failures. Native RGB/depth rendering and the Habitat-Lab
API fixture were also verified on the first HM3D scene.

See [HM3D acceptance and artifact hashes](evidence/navigation/hm3d/acceptance.json).
These are short-range GPS/compass PointNav smoke results. Both selected scenes
come from HM3D minival, so this experiment does not implement official HM3D
training/evaluation splits, a complete unseen-scene benchmark, or language
following. The mounted release lacks semantic annotations; the simulator logs
a missing semantic descriptor, while RGB, depth, navigation and scoring run.

The standard framework command above also accepts the HM3D checkpoint and scene.
Use the first scene with seed 200000 to reproduce the main test, or the second
scene with seed 300000 to reproduce the additional transfer test.

## Use the adapters

```python
from types import SimpleNamespace
import yaml
from vlastudio.benchmark.habitat import create_env

config = yaml.safe_load(open('src/vlastudio/configs/env/habitat/habitat_pointnav.yaml'))
config['scene'] = '/absolute/path/to/skokloster-castle.glb'
env = create_env(SimpleNamespace(**config))
try:
    obs = env.reset()
    # Scalar actions are convenient in a direct loop. Framework policies use MetaAction.
    obs, reward, done, info = env.step(0)
    print(info['success'], info['spl'])
finally:
    env.close()
```

Habitat actions: `0 STOP`, `1 FORWARD`, `2 LEFT`, `3 RIGHT`.
PointNav state is `[distance, bearing]`, RGB is uint8 `(1,3,H,W)` and depth is
float meters `(1,H,W)`. Positive bearing means left. Native navmesh distance
determines success and shortest-path SPL; traveled distance accumulates actual
post-collision movement. Stopped or timed-out episodes remain terminal until reset.

Each navigation module exports its own `evaluate` function because the older
manipulation evaluator treats `done` as success. The navigation evaluator tracks
termination separately, preserves failed results, and freezes finished vector slots.
When constructing multiple R2R environments, set `rollout_index` and
`episode_stride` to nonoverlapping offsets/strides for the desired episode set.

## Continuous R2R / VLN-CE task

Install the same-version Habitat-Lab into the Habitat environment:

```bash
git clone --branch v0.3.3 --depth 1 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
python -m pip install -e third_party/habitat-lab/habitat-lab
```

Use `habitat.habitat_vlnce`, supply the scenes and episode paths using
`habitat_overrides` (Habitat/Hydra dot notation), and select the desired split:

```yaml
habitat_overrides:
  - habitat.simulator.turn_angle=30
  - habitat.dataset.data_path=/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
  - habitat.dataset.scenes_dir=/datasets/scene_datasets
  - habitat.dataset.split=val_unseen
```

Episodes must match Habitat-Lab's `R2RVLN-v1` schema, including
`instruction_vocab`, instruction text, reference path and trajectory id. If a
source release uses another schema, convert and verify those fields before use;
the adapter does not silently fabricate missing reference data. The task's
`success`/`spl` measurements remain authoritative. Instruction text is exposed as
`MetaObs.raw_lang`; missing point-goal observations become zeros rather than
exposing reference goals. Additional registration modules and legacy config
loading are available for explicitly version-matched external task ports.

## Matterport3D / classic R2R

Follow the upstream native build instructions for
[Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator).
Build its Python binding for the same interpreter, expose `build` on `PYTHONPATH`,
and supply authorized Matterport panorama assets. Then use
`matterport3d.matterport_r2r` with:

```yaml
episodes: /datasets/R2R_val_seen.json
dataset_path: /datasets/v1/scans
connectivity_path: /src/Matterport3DSimulator/connectivity
```

R2R records are expanded per instruction, retaining `path_id_instructionIndex`.
Actions add `4 LOOK_UP`, `5 LOOK_DOWN`; FORWARD selects visible navigable index 1
or stays in place on obstruction. This is the classic six-action interface,
not a panoramic variable-candidate policy. Discretized MatterSim turns use +/-1
increments. RGB BGR channels are converted to RGB and depth units to meters.
State contains heading, elevation and visible degree; reference path, teacher,
goal world coordinates and distance-to-goal are withheld from policy inputs.
Scoring uses weighted connectivity-graph shortest paths, explicit STOP within 3m,
and actual graph-edge path length for SPL. No data is downloaded or license
agreement accepted automatically.

Hugging Face hosts [official public Habitat test scenes](https://huggingface.co/datasets/ai-habitat/habitat_test_scenes).
Community collections such as [VLN-CE-R2R for EASI](https://huggingface.co/datasets/oscarqjh/VLN-CE-R2R_easi)
also package annotations and meshes, but are not the original benchmark release;
check their schema, action angles, splits and upstream asset access conditions.
The [original MP3D project](https://niessner.github.io/Matterport/) requires a signed
access request. A repository of rendered images is not a substitute for runnable
scene geometry, and Habitat GLBs are not MatterSim panorama assets.

## Tests and primary sources

```bash
PYTHONPATH=src python -m pytest tests/test_navigation.py -q
```

Tests cover invalid/nonintegral actions, action tags, STOP and timeout failures,
SPL, frozen terminal vector slots, graph distances, MatterSim rotation/channel/
depth conventions, checkpoint loading and framework policy inference.

- [Habitat-Sim v0.3.3 source build](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/BUILD_FROM_SOURCE.md)
- [Habitat-Lab continuous R2R config](https://github.com/facebookresearch/habitat-lab/blob/v0.3.3/habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml)
- [VLN-CE official code and dataset requirements](https://github.com/jacobkrantz/VLN-CE)
- [Matterport3DSimulator API and data setup](https://github.com/peteanderson80/Matterport3DSimulator)
