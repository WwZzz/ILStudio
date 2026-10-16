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
The policy takes RGB, depth and original instruction text, and outputs STOP, FORWARD or LEFT/RIGHT.
[Recorded full-split results and checkpoint provenance](../../../docs/evidence/navigation/standard-interface/README.md).

## Data composition

`VLNCEDataset` reads official `train.json.gz` episodes (scene/start pose, instruction
text) and `train_gt.json.gz` expert action sequences.
It replays the recorded actions in Habitat using the authorized MP3D scenes and
caches the resulting RGB/depth observations. No policy-generated demonstrations
or DAgger collection are needed. One item is an entire instruction trajectory of
length `T` with `sample_mode: episode`, the example default. The dataset owns
temporal sampling and action alignment, following `HDF5SequenceDataset` (FastWAM).
The collator keeps variable-length episodes separate.

Illustrative sample, with arrays shown symbolically:

```python
{
    "raw_lang": "Leave the bedroom and stop by the stairs.",
    "action": [[1], [1], [2], [1], [0]],  # example T=5; expert labels
    "state": empty_state,           # (T,0); no supplied proprioception
    "is_pad": [False] * T,
    "image": rgb,                   # uint8 (1,T,3,224,224)
    "depth": normalized_depth,      # float32 (1,T,256,256), range [0,1]
    "reasoning": {
        "image_type": "rgb",
        "image_is_pad": [False] * T,
        "frame_ids": [0, 1, 2, 3, 4],
        "action_frame_ids": [0, 1, 2, 3, 4],
        "navigation": navigation_metadata,  # T observable control-contract records
        "supervision": {            # training-only measurements, not inputs
            "distance_to_goal": distances,          # (T,)
            "initial_distance_to_goal": initial_distances, # (T,)
        },
    },
    "episode_id": "example_episode",
}
```

The dataset stores raw replay observations in its data cache; it does not load a
policy or checkpoint. CMA's processor tokenizes `raw_lang` with its packaged
2504-word release vocabulary, adapts tensor layouts and computes the auxiliary
progress target. With frozen visual encoders it computes features; the standard
data task cache persists these processed samples. The processor owns no cache
directory. Changing encoder weights invalidates this processed cache while raw
RGB/depth remain reusable. Trainable encoders receive RGB/depth tensors instead.

For fixed windows set dataset args `sample_mode: window`, `chunk_size: 32`.
By default this loads 32 observations and their 32 actions. `image_offsets`
selects observation offsets independently, as in FastWAM; CMA requires one
observation per action. Windows never cross episodes: final frames/actions are
repeated with `reasoning.image_is_pad`/`is_pad` marking padding. A single observation has
`image` `(1,3,H,W)`; multiple observations have `(1,T,3,H,W)`. Whole episodes
preserve recurrent history from reset and are the baseline training default.

Action IDs are `0=STOP`, `1=MOVE_FORWARD`, `2=TURN_LEFT`, `3=TURN_RIGHT`.
The configured primitives move 0.25 m or rotate 15 degrees. At evaluation the
environment returns current RGB/depth, `raw_lang`, an empty state and a timestep.
Additional control-contract metadata lives in `reasoning.navigation`.
The learned VLN configuration does not expose goal coordinates, expert actions,
progress, annotation tokens or the reference path. The policy owns its
recurrent memory and previous-action history.
