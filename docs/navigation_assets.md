# Navigation assets

Use existing authorized local or mounted data. Scene geometry, panoramas,
annotations, visual features and model weights are separate assets. This example
does not download MP3D scenes or accept data terms automatically.

## Data layout

Set data paths in the selected simulator folder's `config/`. A typical layout is:

```text
/datasets/
  scene_datasets/mp3d/<scan>/<scan>.glb    # Habitat geometry and matching navmesh
  R2R_VLNCE_v1-3_preprocessed/
    train/train.json.gz
    train/train_gt.json.gz
    val_unseen/val_unseen.json.gz
  R2R/
    R2R_train.json
    R2R_val_unseen.json
```

For native MatterSim image rendering, provide its preprocessed RGB skyboxes under
`<scan>/matterport_skybox_images`. The VLN-BERT paper config instead uses cached
panorama features and sets `rendering_enabled: false`, `depth_enabled: false`.
Habitat GLBs alone do not supply MatterSim skybox images.

Official download instructions: [VLN-CE](https://github.com/jacobkrantz/VLN-CE/tree/729d141b2ee10628061ada74dd3a5b9f70faeba5),
[Recurrent VLN-BERT data and weights](https://github.com/YicongHong/Recurrent-VLN-BERT/tree/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e).
The recorded QZ run used `internnav:scene-n1`, mounted at
`/inspire/dataset/internnav/scene-n1`; use this directory for `habitat.dataset.scenes_dir`.
Availability depends on the instance's dataset mounts.

## CMA checkpoint

Place the authors' `cma-pm-da-aug.pth` under `assets/`, then convert it:

```bash
python examples/09_navigation/habitat/prepare_weights.py \
  --source assets/cma-pm-da-aug.pth \
  --sha256 06278fdbb52ad8c453b404dacb079fb353252056314b1c567a381df77e6827a3 \
  --output checkpoints/example09/cma-official
```

Conversion verifies the source hash and uses restricted tensor loading, preserving
the original checkpoint. The converted directory contains the model and framework
metadata required by evaluation and fine-tuning.

## Recurrent VLN-BERT checkpoint and features

Prepare the authors' `ResNet-152-places365.zip`, the PREVALENT checkpoint
`snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen` from their official
`snap.zip`, and the BERT uncased `vocab.txt`. Convert features before the model:

```bash
python examples/09_navigation/mattersim/prepare_features.py \
  --archive assets/ResNet-152-places365.zip \
  --sha256 791344c4987de20ff792f80159a7f9e229310f421f59e4494c564b938c5bab8d \
  --output data/example09_places365

python examples/09_navigation/mattersim/prepare_weights.py \
  --source assets/snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen \
  --sha256 612fccc817d104ef09de4df70c081ddd692cbb85c219785c7bbfc627507c3ee0 \
  --vocabulary assets/vocab.txt --feature-store data/example09_places365 \
  --output checkpoints/example09/vlnbert-official
```

The indexed feature store contains 10,567 panoramas, each 36×2,048 float32 values.
The model config records its absolute feature-store path; update `feature_store`
in the converted model's `config.json` if you move that store. Conversion also
checks the BERT vocabulary hash and strictly loads all navigator tensors.

Source hashes and runtime details are retained in the
[recorded acceptance evidence](evidence/navigation/paper/README.md).
Return to [Example 09](../examples/09_navigation/README.md) to train and evaluate.
