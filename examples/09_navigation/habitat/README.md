# Habitat — CMA on VLN-CE

Run from the repository root on Linux with Python 3.10 and an NVIDIA GPU.

## Setup

Install [Habitat-Sim and Habitat-Lab](../../../docs/navigation.md#install-habitat-sim-on-python-310),
then prepare the [MP3D data and CMA weights](../../../docs/navigation_assets.md#cma-checkpoint).
Set dataset paths in [`config/`](config/).

```bash
python -m pip install -e . 'gym==0.23.0' 'yacs==0.1.8'
```

## Evaluate released weights

```bash
python examples/09_navigation/habitat/evaluate.py
```

## Train and evaluate

```bash
python examples/09_navigation/habitat/train.py
python examples/09_navigation/habitat/evaluate.py \
  --model checkpoints/example09/cma-trained/epoch-4 \
  --output results/example09_cma_trained
```

The policy receives RGB, depth and instruction tokens, and outputs STOP,
FORWARD or LEFT/RIGHT. Training continues from the released weights on 256 routes.

## Results

Full `val_unseen`: 1,839 instruction episodes, 11 scenes.

| Checkpoint | SR | SPL |
|---|---:|---:|
| Author-corrected reference | 29% | 27% |
| Released weights via VLAStudio | 30.13% | 27.98% |
| VLAStudio continuation | **29.85%** | **27.72%** |

For the public-scene PointNav sample, see the [quick start](../../../docs/navigation.md#train-and-accept-a-simple-policy).
Detailed protocols and checkpoint hashes are in the [acceptance report](../../../docs/evidence/navigation/paper/README.md).
