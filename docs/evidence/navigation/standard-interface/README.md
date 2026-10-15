# Navigation through the standard VLAStudio interfaces

Both Example 09 training entries now use `vla.load_dataset`, `vla.load_policy`
and `vla.train`. The following checks were run on real MatterSim and Habitat,
using the normal training worker, optimizer, scheduler and checkpoint saving.

## Training and reload

Each model started from the immutable official checkpoint and completed two
optimizer updates on one original training route with its three instructions.
These short continuations validate the standard interface; they do not reproduce
the papers' full training schedules. The earlier longer continuation experiments
remain recorded in [the paper acceptance report](../paper/README.md).

| Policy | Updates | Parameter change, L2 | Strict checkpoint reload |
|---|---:|---:|---|
| CMA | 2 | 0.002851024 | Identical saved tensors |
| Recurrent VLN-BERT | 2 | 0.001640108 | Identical saved tensors |

[training-proof.json](training-proof.json) records losses, checkpoint hashes and
parameter changes. The original training instructions and paths supply the
supervision. No DAgger collection is needed. The R2R cache used in this training
was compared tensor by tensor with replay of the exact original reference route;
all three instruction samples matched, including actions, labels, features and
states ([audit](reference-replay-audit.json)).

The proof uses seed 20261015, batch size 1, gradient accumulation 1,
`max_steps=2`, constant learning rate, AdamW, and checkpoint saving every step.
Learning rates are 1e-6 for CMA and 1e-7 for VLN-BERT. Example defaults use larger
route subsets and epoch schedules, configured in each simulator's training YAML.

## Full validation of these saved checkpoints

| Simulator / policy | Split | Episodes / scenes | SR (%) | SPL (%) | NE (m) | TL (m) |
|---|---|---:|---:|---:|---:|---:|
| Habitat / CMA | val_unseen | 1,839 / 11 | 30.0707 | 27.9531 | 7.3847 | 8.3394 |
| MatterSim / Recurrent VLN-BERT | val_unseen | 2,349 / 11 | 62.7501 | 56.8418 | 3.9263 | 12.0124 |

CMA exceeds the user-approved author-corrected reference of SR 29%, SPL 27%.
VLN-BERT matches the paper's SR 63%, SPL 57% at the published whole-percentage
precision. These are complete validation splits, not hidden-test estimates.
The scoring and simulator-version qualifications in the earlier report continue
to apply: CMA requires STOP within 3 m; R2R uses the final viewpoint within 3 m,
with 15 greedy decisions and the published Places365 panorama features.

[audit.json](audit.json) independently recomputes every aggregate from the saved
episode journals, verifies exact split membership with no duplicates or missing
episodes, and checks the evaluated checkpoint hashes. Each model folder contains
its summary, evaluation configuration and compressed episode journal. No model
weights or MP3D scene files are stored in Git.

## Data, inference and runtime

- Standard dataset classes download annotations and features lazily; policy
  loaders download and convert configured pretrained weights. Cached assets are
  reused. [download-proof.json](download-proof.json) records an actual R2R fallback
  download and offline reuse. The pinned public mirror was verified against the
  complete official records of all three splits.
- MP3D geometry uses the existing `internnav:scene-n1` mount. It is not copied or
  republished. Native MatterSim executes graph actions; Habitat renders RGBD.
- Standard `vla.load_env(...).evaluate` also completed six held-out instruction
  episodes per saved model, exercising the shared inference workers and action
  transport. Those small checks are pipeline diagnostics, not benchmark estimates
  ([inference-proof.json](inference-proof.json)).
- [runtime.json](runtime.json) records the isolated standard-training runtime.
  The original benchmark environment and official checkpoints remain unchanged.
- Regression tests: **99 passed, 2 skipped**. The skipped cases require optional
  FastAPI HTTP transport dependencies; [tests.log](tests.log) records the result.

[manifest.json](manifest.json) contains SHA-256 checksums for the archived evidence.
`${NAVIGATION_ROOT}` replaces the private experiment directory in text artifacts.
