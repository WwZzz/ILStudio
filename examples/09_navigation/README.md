# 09 — Navigation

Train and evaluate navigation policies in Habitat and MatterSim. Run from the
repository root on Linux with Python 3.10; the learned VLN models use an NVIDIA GPU.

| Sample | Task / environment | Policy input | Policy output |
|---|---|---|---|
| Quick start | PointNav / Habitat-Sim | GPS/compass goal distance and bearing | STOP, FORWARD, LEFT, RIGHT |
| Continuous VLN | R2R VLN-CE / Habitat-Lab | RGB, normalized depth, instruction tokens | STOP, FORWARD, LEFT, RIGHT |
| Viewpoint VLN | R2R / MatterSim | Instruction, current-viewpoint panorama features, candidate geometry | STOP or candidate viewpoint ID |

R2R (Room-to-Room) pairs routes through Matterport3D scenes with natural-language
instructions. Its original simulator moves between discrete viewpoints; VLN-CE
uses continuous movement in scene geometry. Candidate generation can be exposed
by the environment; the policy ranks candidates and chooses an action. The
environment config explicitly declares the observation and action interface.

## Setup

Create a separate environment and install the shared Python dependencies:

```bash
python -m venv .venv-navigation
source .venv-navigation/bin/activate
python -m pip install -e . 'torch==2.4.1' 'torchvision==0.19.1' \
  'transformers==4.50.0' 'gym==0.23.0' 'yacs==0.1.8' \
  'cmake>=3.24,<3.28' ninja 'setuptools<70' wheel
```

For Habitat, follow the [Habitat-Sim source build](../../docs/navigation.md#install-habitat-sim-on-python-310).
For CMA, also install [Habitat-Lab 0.3.3](../../docs/navigation.md#continuous-r2r--vln-ce-task).
For MatterSim on Ubuntu 22.04, run the installer, then activate its native libraries:

```bash
python examples/09_navigation/install_mattersim.py --prefix "$PWD/.mattersim"
source "$PWD/.mattersim/activate.sh"
```

The installer builds MatterSim for the active Python in a private prefix. Native
simulators are prepared explicitly; importing VLAStudio does not install them.
See [setup details and host prerequisites](../../docs/navigation.md).

## Quick start: train PointNav on a public scene

After installing Habitat-Sim:

```bash
python examples/09_navigation/download_test_scene.py
python examples/09_navigation/train_and_evaluate.py \
  --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb \
  --output results/example09_pointnav
```

This trains a small MLP with behavior cloning and DAgger. It selects a checkpoint
on validation episodes and evaluates 100 held-out episodes against random and
immediate-STOP baselines. Acceptance requires SR >=30% and above random. The
recorded run reached **87% SR, 86.23% SPL**. This GPS/compass sample does not use
language; the following two samples demonstrate learned instruction following.

The output includes `policy.pt`, episode definitions, metrics, training history
and a rendered rollout. A navmesh-only `--no-render` run is also available.

## Prepare VLN assets and configs

Provide original R2R and VLN-CE annotations, MP3D geometry for Habitat, official
model weights, and the authors' Places365 feature archive plus BERT vocabulary
for VLN-BERT. Data may stay on an existing mounted directory. See
[asset preparation](ASSETS.md) for conversion commands and source checksums.

Generate four resolved environment YAMLs, replacing the paths with your own:

```bash
python examples/09_navigation/prepare_configs.py \
  --vlnce-data /datasets/R2R_VLNCE_v1-3_preprocessed \
  --scenes-dir /datasets/scene_datasets \
  --r2r-data /datasets/R2R \
  --mattersim-prefix "$PWD/.mattersim" \
  --output configs/example09
```

`--scenes-dir` contains `mp3d/<scan>/<scan>.glb` and matching navigation meshes.
`--r2r-data` contains `R2R_train.json` and `R2R_val_unseen.json`.
The generated `cma-{train,val-unseen}.yaml` and `r2r-{train,val-unseen}.yaml`
retain the packaged paper sensor/action settings and use absolute asset paths.
The generator prepares YAML without loading simulators or downloading assets.

The R2R paper config disables native image rendering because its policy reads
official cached features at the **current** viewpoint. MatterSim still executes
every turn and move on the real graph. To use native skybox rendering in a custom
policy, provide `--matterport-scans`, enable rendering in your config and match
the policy's visual inputs. See [candidate configuration](../../docs/navigation.md#candidate-observations-and-action-configuration).

## Continuous VLN: CMA on Habitat

Evaluate converted official weights, then continue training on the train split:

```bash
python examples/09_navigation/evaluate_cma.py \
  --config configs/example09/cma-val-unseen.yaml \
  --model checkpoints/example09/cma-official --episodes 1839 --batch-size 8 \
  --output results/example09_cma_official

python examples/09_navigation/finetune_cma.py \
  --config configs/example09/cma-train.yaml \
  --ground-truth /datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --model checkpoints/example09/cma-official \
  --trajectories 256 --epochs 4 --batch-size 5 --unroll 32 \
  --lr 1e-6 --seed 20261015 --output checkpoints/example09/cma-trained

python examples/09_navigation/evaluate_cma.py \
  --config configs/example09/cma-val-unseen.yaml \
  --model checkpoints/example09/cma-trained/epoch-4 --episodes 1839 --batch-size 8 \
  --output results/example09_cma_trained
```

CMA receives RGB 224×224, normalized depth 256×256 and original instruction
tokens. Actions are `0 STOP`, `1 FORWARD` (0.25 m), `2 LEFT` / `3 RIGHT` (15°).
Success requires explicit STOP within 3 m, with a 500-action limit.

Training checks native expert trajectory replay before learning and uses
inflection weighting plus progress supervision. Optional
[`collect_cma_dagger.py`](collect_cma_dagger.py) collects labels at learner-visited
training states; pass its feature directory to `finetune_cma.py --precollected`
with the same model, config, trajectory count and seed. Stored previous actions
are the actions actually executed, separate from expert supervision labels.

## Viewpoint VLN: Recurrent VLN-BERT on MatterSim

Activate MatterSim's native libraries in each new terminal before these commands:

```bash
source "$PWD/.mattersim/activate.sh"
python examples/09_navigation/evaluate_vlnbert.py \
  --config configs/example09/r2r-val-unseen.yaml \
  --model checkpoints/example09/vlnbert-official --batch-size 8 \
  --output results/example09_vlnbert_official

python examples/09_navigation/finetune_vlnbert.py \
  --config configs/example09/r2r-train.yaml \
  --model checkpoints/example09/vlnbert-official \
  --trajectories 512 --epochs 2 --batch-size 4 --lr 1e-7 --seed 20261015 \
  --output checkpoints/example09/vlnbert-trained

python examples/09_navigation/evaluate_vlnbert.py \
  --config configs/example09/r2r-val-unseen.yaml \
  --model checkpoints/example09/vlnbert-trained/epoch-1 --batch-size 8 \
  --output results/example09_vlnbert_trained
```

The model uses instruction text, 36-view Places365 features at the current
viewpoint and candidate geometry. Output `0` means STOP; other action IDs select
from the observation's current candidate list. The config declares
`action_mode: candidate_viewpoint`, `candidate_source: simulator`,
`candidate_scope: all_neighbors` and `candidate_view_alignment: panoramic`.
The simulator turns toward the selected source view before moving to its neighbor.

Published R2R scores the final viewpoint within 3 m, including at the 15-decision
limit, so the config declares `success_requires_stop: false`. The environment also
records `explicit_stop_success`. The model receives no goal, expert action,
reference route or success metric. The short training run uses train-only
teacher-forced imitation learning; the recorded best checkpoint was epoch 1.

## Recorded results

Both full evaluations use 11 unseen scenes and greedy inference without search.

| Environment / model | Full validation episodes | Official weights SR / SPL | Trained weights SR / SPL | Reference SR / SPL |
|---|---:|---:|---:|---:|
| Habitat / CMA | 1,839 | 30.13% / 27.98% | **29.85% / 27.72%** | Author-corrected **29% / 27%** |
| MatterSim / Recurrent VLN-BERT | 2,349 | 62.75% / 56.84% | **62.58% / 56.68%** | Paper **63% / 57%**, whole-percentage precision |

These trained weights are short continuations from official pretrained weights.
The full original training schedules were not reproduced. CMA's original paper
SPL 30% is not claimed. The runtime uses Habitat 0.3.3, compared with the original
CMA build's 0.1.7. All completed comparisons, training updates, checkpoint hashes,
full episode journals and the 51 passing regression checks are documented in the
[acceptance report](../../docs/evidence/navigation/paper/README.md).
The exact comparison protocol is in [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md).

Evaluators save `run.json`, `episodes.jsonl` and `evaluation.json`. Repeating an
evaluation command resumes only with matching model/config hashes. Training saves
manifests, losses and separate epoch checkpoints; official checkpoints remain intact.
For shared inference-worker examples and real-asset diagnostic checks, see
[the navigation guide](../../docs/navigation.md).
