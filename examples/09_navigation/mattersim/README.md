# MatterSim — Recurrent VLN-BERT on R2R

Run from the repository root on Ubuntu 22.04 with Python 3.10 and an NVIDIA GPU.

## Setup

```bash
python -m pip install -e . 'cmake>=3.24,<3.28'
python -m vlastudio.benchmark.matterport3d.install --prefix "$PWD/.mattersim"
source "$PWD/.mattersim/activate.sh"
```

Standard R2R annotations, visual features and configured pretrained weights download and cache automatically.
Use `config/task.yaml` to select existing data or change paths.

## Train

```bash
python examples/09_navigation/mattersim/train.py
```

The script uses `vla.load_dataset`, `vla.load_policy` and `vla.train`.
For a short run, add `--steps 2`. Change training settings in `config/training.yaml`.

## Evaluate

```bash
python examples/09_navigation/mattersim/evaluate.py
python examples/09_navigation/mattersim/evaluate.py \
  --model checkpoints/example09/vlnbert --output results/example09_vlnbert_trained
```

The first command loads released weights automatically. The second uses the trained checkpoint.
The policy takes instructions, current-viewpoint features and candidate geometry, then selects STOP or a candidate.
[Recorded full-split results and checkpoint provenance](../../../docs/evidence/navigation/standard-interface/README.md).
