# Navigation data and pretrained weights

Example 09 uses the standard VLAStudio dataset and training interfaces:

```python
import vlastudio as vla

dataset = vla.load_dataset("navigation.r2r")  # or navigation.vlnce
policy = vla.load_policy("recurrent_vlnbert")  # or vlnce_cma
vla.train(policy, dataset, "path/to/training.yaml", output_dir="checkpoints/navigation")
```

`load_dataset` creates a configuration handle. When training materializes the
configured dataset, it downloads missing standard annotations/features and builds
its replay cache. `load_model` downloads and converts weights only when
`pretrained_weight_path` is configured. No separate preparation command is needed.

## Datasets

- `R2RDataset`: original R2R route/instruction records, all instructions per selected
  route, real MatterSim graph replay and current-viewpoint Places365 features.
- `VLNCEDataset`: original preprocessed VLN-CE annotations and `train_gt` expert
  actions, real Habitat RGBD replay, and optional frozen-encoder feature caching.

Both implement the standard dataset contract, including `action`, `state`,
`is_pad`, `raw_lang` and episode IDs. R2R exposes stable simulator action IDs in
`action`; separate `labels` index the candidate tensor for classification loss.
Training labels and progress supervision never enter inference observations.

Datasets accept `dataset_dir`, `cache_dir`, `download` and `max_trajectories`.
R2R also accepts `connectivity_path` and `feature_store`; VLN-CE accepts
`scenes_dir` and `feature_checkpoint`. The CMA preset caches features with the
same frozen official encoder as its pretrained policy. Set `feature_checkpoint`
to null to use raw RGBD instead. Pretrained encoder choice must match the policy.

MP3D geometry uses the existing authorized scene directory or mount; it is not
bundled or republished. On the recorded QZ instance, the scene root is
`/inspire/dataset/internnav/scene-n1`, containing `mp3d/<scan>/<scan>.glb` and navmeshes.
The published VLN-BERT feature protocol does not require rendering skybox images.

R2R annotations use the [official source](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
If Dropbox is unreachable, the loader uses a
[pinned public metadata mirror](https://huggingface.co/datasets/thomas-yanxin/r2r-metadata/tree/6fe3b2d84de51c6e788fdc361d4f160728228a79),
checking both file hashes and exact semantic equality with the original ordered
records. VLN-CE uses the [authors' preprocessed release](https://github.com/jacobkrantz/VLN-CE).
Existing standard datasets suffice; this workflow does not require uploading another copy.

## Models and caches

`pretrained_weight_path` accepts `official`, a local converted checkpoint or a
Hugging Face model repository. The loader checks published source hashes and uses
restricted tensor loading for conversion. Null initializes the model without
fetching pretrained weights; random VLN-BERT initialization requires a local BERT
`vocabulary` path. Set `download: false` to require local assets.

`VLASTUDIO_CACHE` controls the shared asset/model cache. The dataset-only
`cache_dir` passed to `vla.load_dataset` controls downloaded data and replay caches.
Model checkpoints retain the resolved feature-store path for independent inference.

DAgger is optional online expert relabeling at learner-visited states. Standard
expert demonstrations are available, so Example 09 uses offline supervised
training without a separate collection step. Optional diagnostics and PointNav
scripts remain under `scripts/navigation`, outside the example.

[Example 09](../examples/09_navigation/README.md) ·
[Standard-interface training and full benchmark evidence](evidence/navigation/standard-interface/README.md)
