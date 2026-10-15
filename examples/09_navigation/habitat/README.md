# Habitat — CMA on VLN-CE

Run from the repository root on Linux with Python 3.10 and an NVIDIA GPU.

## Setup

Install [Habitat-Sim and Habitat-Lab](../../../docs/navigation.md#install-habitat-sim-on-python-310), then:

```bash
python -m pip install -e . 'gym==0.23.0' 'yacs==0.1.8'
```

Set the MP3D scene root in `config/task.yaml` and `config/val_unseen.yaml`.
Standard annotations and configured pretrained weights download and cache automatically.

## Train

```bash
python examples/09_navigation/habitat/train.py
```

The script uses `vla.load_dataset`, `vla.load_policy` and `vla.train`.
For a short run, add `--steps 2`. Change training settings in `config/training.yaml`.

## Evaluate

```bash
python examples/09_navigation/habitat/evaluate.py
python examples/09_navigation/habitat/evaluate.py \
  --model checkpoints/example09/cma --output results/example09_cma_trained
```

The first command loads released weights automatically. The second uses the trained checkpoint.
The policy takes RGB, depth and instruction tokens, and outputs STOP, FORWARD or LEFT/RIGHT.
[Recorded full-split results and checkpoint provenance](../../../docs/evidence/navigation/standard-interface/README.md).
