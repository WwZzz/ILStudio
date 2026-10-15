# MatterSim — Recurrent VLN-BERT on R2R

Run from the repository root on Ubuntu 22.04 with Python 3.10 and an NVIDIA GPU.

## Setup

```bash
python -m pip install -e . 'cmake>=3.24,<3.28'
python examples/09_navigation/mattersim/install.py --prefix "$PWD/.mattersim"
source "$PWD/.mattersim/activate.sh"
```

Prepare the [R2R data, features and weights](../../../docs/navigation_assets.md#recurrent-vln-bert-checkpoint-and-features),
then set dataset paths in [`config/`](config/). Activate MatterSim in each new terminal.

## Evaluate released weights

```bash
python examples/09_navigation/mattersim/evaluate.py
```

## Train and evaluate

```bash
python examples/09_navigation/mattersim/train.py
python examples/09_navigation/mattersim/evaluate.py \
  --model checkpoints/example09/vlnbert-trained/epoch-1 \
  --output results/example09_vlnbert_trained
```

The policy receives instructions, current-viewpoint features and candidate geometry,
and selects STOP or a candidate viewpoint. Training continues from released weights on 512 routes.

## Results

Full `val_unseen`: 2,349 instruction episodes, 11 scenes.

| Checkpoint | SR | SPL |
|---|---:|---:|
| Paper (whole-percentage precision) | 63% | 57% |
| Released weights via VLAStudio | 62.75% | 56.84% |
| VLAStudio continuation | **62.58%** | **56.68%** |

Detailed protocols and checkpoint hashes are in the [acceptance report](../../../docs/evidence/navigation/paper/README.md).
