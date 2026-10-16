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

## Data composition

`R2RDataset` combines official R2R annotations (scan, start heading, viewpoint
path and usually three human instructions per path), connectivity graphs and the
released Places365 ResNet-152 panorama features. A viewpoint has 36 camera views
(12 headings × 3 elevations), each with 2048 visual features. Each instruction
becomes a separate item; `max_trajectories` selects paths before expanding their
instructions. One item is a complete sequence of length `T`, up to the configured
15-step reference protocol. Replay uses MatterSim's graph without RGB rendering;
the learned baseline reads cached visual features.

Illustrative sample, with arrays shown symbolically:

```python
{
    "raw_lang": "Walk through the door and stop beside the stairs.",
    "candidate_features": features,  # (T,N,2176): 2048 visual + 128 angle features
    "candidate_mask": valid,         # bool (T,N), includes valid STOP rows
    "angles": angles,                # (T,128), current orientation encoding
    "labels": [1, 0, 2],              # example per-step candidate indices
    "action": [[17], [23], [0]],      # example simulator IDs; 0 is STOP
    "state": states,                 # (T,3): heading, elevation, native visible degree
    "is_pad": [False] * T,
    "episode_id": "example_path_instruction",
}
```

`N` is padded to the item's largest candidate count plus a STOP row. Each step's
STOP row follows that step's valid movement candidates; remaining rows are masked.
`labels` are indices into this per-step feature list, while `action` contains
simulator IDs. These are different number spaces: a movement action ID is stable
within a scan, not simply the candidate's list position. The final STOP index in
the illustrative example is 2, meaning two movement candidates at that step.

At evaluation, the environment supplies original text, current viewpoint identity
and candidate metadata (action ID, neighbor identity, relative angle/distance and
source panorama view). The policy retrieves cached features and tokenizes text.
`candidate_source: simulator`, `candidate_scope: all_neighbors` and
`candidate_view_alignment: panoramic` make this contract explicit in the config.
Candidates are reachable neighbors, not teacher choices. Reference paths, goal
viewpoint and correct candidate labels are withheld from policy inputs.
