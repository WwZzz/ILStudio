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
instructions. With `sample_mode: episode`, one item is its complete annotated
sequence of length `T`, including the final STOP. Dataset-owned temporal sampling
follows `HDF5SequenceDataset` (FastWAM). `visual_mode: features` uses MatterSim's
graph without RGB rendering and reads the published visual data from a shared
data-layer feature store.

Illustrative sample, with arrays shown symbolically:

```python
{
    "raw_lang": "Walk through the door and stop beside the stairs.",
    "visual_features": panoramas,    # float32 (T,36,2048), unmodified source data
    "visual_feature_source": "places365-resnet152-36x2048-v1",
    "navigation": [                 # one metadata record per observation
        {"scan_id": "scan", "viewpoint_id": "viewpoint",
         "candidate_view_alignment": "panoramic",
         "candidates": [            # observable neighbors, not teacher choices
             {"action_id": 17, "viewpoint_id": "neighbor",
              "source_view_index": 12, "relative_heading": 0.2,
              "elevation": 0.0},
         ]},  # ... remaining timesteps
    ],
    "action": [[17], [23], [0]],      # example simulator IDs; 0 is STOP
    "state": states,                 # (T,3): heading, elevation, native visible degree
    "is_pad": [False] * T,
    "image_is_pad": [False] * T,
    "episode_id": "example_path_instruction",
}
```

The dataset does not emit model tokens, angle embeddings, candidate masks or
classification labels. The VLN-BERT processor selects source panorama views,
adds the model's 128-D angle encoding, appends STOP and creates padding masks.
It maps expert simulator action IDs to indices in that per-step candidate list.
Movement action IDs are stable within a scan and are not candidate list positions.
Original 2048-D source features remain data; their download, indexing and mmap
storage belong to the data layer.

For rendered observations choose `visual_mode: rgb` and provide authorized MP3D
skybox assets in `dataset_path`; the dataset then returns uint8 `image`
`(1,T,3,H,W)` instead of `visual_features`. VLN-BERT's published baseline uses
the feature mode; a pixel-based policy can use the RGB mode.

Fixed windows use `sample_mode: window`, `chunk_size: 32`, with optional
`image_offsets`, as in FastWAM. Windows stay inside an episode and repeat final
frames/actions with padding masks. The included recurrent processor requires
one observation per action and removes padded timesteps before computing loss.
Its default is full episodes to retain history from reset. For one RGB frame,
`image` is `(1,3,H,W)`; for multiple frames it is `(1,T,3,H,W)`.

At evaluation, the environment supplies original text, current viewpoint identity
and candidate metadata (action ID, neighbor identity, relative angle/distance and
source panorama view). A shared data reader supplies published panorama features;
the policy adapts them and tokenizes text.
`candidate_source: simulator`, `candidate_scope: all_neighbors` and
`candidate_view_alignment: panoramic` make this contract explicit in the config.
Candidates are reachable neighbors, not teacher choices. Reference paths, goal
viewpoint and correct candidate labels are withheld from policy inputs.
