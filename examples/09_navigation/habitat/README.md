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

## Data composition

`VLNCEDataset` reads official `train.json.gz` episodes (scene/start pose, instruction
text and original vocabulary tokens) and `train_gt.json.gz` expert action sequences.
It replays the recorded actions in Habitat using the authorized MP3D scenes and
caches the resulting RGB/depth observations. No policy-generated demonstrations
or DAgger collection are needed. One item is an entire instruction trajectory of
length `T`; the training collator pads trajectories and constructs sequence masks.

Illustrative sample, with arrays shown symbolically:

```python
{
    "raw_lang": "Leave the bedroom and stop by the stairs.",
    "instruction": token_ids,       # original annotation vocabulary; (L,)
    "action": [[1], [1], [2], [1], [0]],  # example T=5; expert labels
    "state": zeros,                 # (T,2), placeholder; no goal distance/bearing
    "is_pad": [False] * T,
    "rgb_features": rgb_features,   # (T,...), cached RGB encoder output
    "depth_features": depth_features, # (T,...), cached depth encoder output
    "progress": expert_progress,    # (T,), auxiliary training target only
    "episode_id": "example_episode",
}
```

With `feature_checkpoint: official` (the example default), visual features are
computed using that checkpoint and cached; training uses these features, rather
than recomputing/updating the visual encoders on every batch. Without a feature
checkpoint the item instead contains `rgb` `(T,H,W,3)` and `depth` `(T,H,W,1)`.
Changing annotation/config/encoder provenance creates a different replay cache.

Action IDs are `0=STOP`, `1=MOVE_FORWARD`, `2=TURN_LEFT`, `3=TURN_RIGHT`.
The configured primitives move 0.25 m or rotate 15 degrees. At evaluation the
environment returns current RGB/depth, `raw_lang`, original instruction tokens in
`navigation`, and a timestep. The learned VLN configuration does not expose goal
coordinates, expert actions, progress or the reference path. Its two-value state
is a compatibility placeholder, not a navigation target. The policy owns its
recurrent memory and previous-action history.
