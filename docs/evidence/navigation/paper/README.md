# Navigation acceptance — 2026-10-15

Both navigation integrations pass the agreed full-validation acceptance criteria.
The user explicitly selected the authors' corrected CMA reference, SR 29% and
SPL 27%, on 2026-10-15. The original CMA paper's SPL 30% is **not claimed**.
For Recurrent VLN-BERT, the primary SR/SPL comparison uses the paper's published
whole-percentage precision. These are validation results, not hidden-test results.

| Environment / model | Checkpoint | Episodes / scenes | SR (%) | SPL (%) | NE (m) | TL (m) |
|---|---|---:|---:|---:|---:|---:|
| Habitat / CMA | Author-corrected reference | — | 29 | 27 | 7.60 | 8.27 |
| Habitat / CMA | Immutable official checkpoint, reevaluated | 1,839 / 11 | 30.1251 | 27.9798 | 7.3743 | 8.3216 |
| Habitat / CMA | **Selected continuation, epoch 4** | **1,839 / 11** | **29.8532** | **27.7167** | **7.4710** | **8.2400** |
| MatterSim / Recurrent VLN-BERT | Paper, PREVALENT, val_unseen | — | 63 | 57 | 3.93 | 12.01 |
| MatterSim / Recurrent VLN-BERT | Immutable official checkpoint, reevaluated | 2,349 / 11 | 62.7501 | 56.8405 | 3.9269 | 12.0093 |
| MatterSim / Recurrent VLN-BERT | **Selected continuation, epoch 1** | **2,349 / 11** | **62.5798** | **56.6772** | **3.9531** | **12.0548** |

CMA continuation exceeds the corrected SR/SPL reference by 0.8532 / 0.7167
percentage points. VLN-BERT continuation rounds to the paper's 63% / 57%; its
unrounded differences from those printed values are −0.4202 / −0.3228 percentage
points. Its NE and TL differ by +0.0231 m / +0.0448 m. Official VLN-BERT
reevaluation matches all four paper metrics at their printed precision.
Neither short continuation improves on its immutable official source checkpoint.

## What was trained

These are actual train-only continuation experiments from the authors' weights,
not from-scratch reproductions of the complete original training schedules.
The checkpoints have distinct hashes and measured nonzero parameter changes.

| Selected model | Training routes | Training | Updates at selected checkpoint | Parameter change, L2 |
|---|---:|---|---:|---:|
| CMA | 256, covering all 61 training scenes | Teacher-forced imitation, progress supervision, LR 1e-6, 4 epochs | 630 | 0.1057972912 |
| Recurrent VLN-BERT | 512 | Teacher-forced imitation, LR 1e-7, selected epoch 1 | 128 | 0.0156836051 |

The VLN-BERT lineage ran two epochs / 256 updates; the selected epoch-1 checkpoint
contains 128 updates. No augmentation or RL was used in this short continuation.
CMA training first verifies native expert trajectory replay; the selected
training cache reached 100% replay success. Privileged expert actions, paths,
goals and progress labels are used only for supervision, outside inference input.

All completed comparisons are retained: CMA LR 1e-5 epochs 1–4, CMA LR 1e-6
epoch 4, VLN-BERT LR 1e-6 epochs 1–2, and VLN-BERT LR 1e-7 epochs 1–2.
Selection uses maximum full `val_unseen` SPL among evaluated continuation
checkpoints. Consequently this is validation-based model selection; there is no
claim of a separate untouched test estimate. The tiny DAgger collection/training
gate checks implementation only and is not part of these benchmark scores.

## Environment and interface checks

- Habitat-Lab and Habitat-Sim 0.3.3, Python 3.10.14, PyTorch 2.4.1, CUDA 12.1,
  RTX 4090. The original CMA runtime was Habitat 0.1.7; the version difference is
  material and is disclosed rather than treated as an identical original build.
- MatterSim source `589d091b111333f9e9f9d6cfd021b2eb68435925`; the installer
  bundles pybind11 2.13.6. `runtime.json` also records the separately installed
  Python pybind11 package, 2.13.5.
- The platform's `internnav:scene-n1` mount contains all 90 MP3D scenes. Habitat
  renders real scene RGB/depth. The R2R paper protocol reads the authors' verified
  36-view Places365 features at the current viewpoint while native MatterSim
  executes turns and moves. Native RGB rendering was checked separately; cached
  features do not supply images from future candidate viewpoints.
- CMA inference receives RGB, normalized depth and original instruction tokens.
  VLN-BERT receives instruction text, current-viewpoint features and candidate
  geometry. Configuration declares candidate alignment, feature mode and action
  semantics. CMA requires explicit STOP within 3 m; published R2R scores the final
  viewpoint, including at its 15-decision limit.
- Both selected full journals have exact original split membership and unique
  episode indices. Training IDs belong to the original train annotations and
  training scenes are disjoint from the 11 validation scenes. Every full journal
  was independently reduced again to verify SR, SPL, NE and TL: **11 complete
  evaluations**, including the official checkpoints and all trained comparisons.
- Official model conversion strictly loads all 520 CMA / 337 VLN-BERT source
  tensors using restricted tensor loading. CMA's additional zero batch-counter
  buffer has no evaluation effect. Original official checkpoints remain intact.
- The shared framework inference path ran native simulator rollouts for both
  models. VLN-BERT's six checked episodes have exactly the same per-episode
  SR/SPL/NE/TL as the full paper evaluator, with maximum metric difference 0.
- Final regression check: **51 tests passed**, covering navigation contracts,
  candidate geometry/action mapping, model inputs, recurrent resets and configs.

## Saved checkpoints and evidence

Set `NAVIGATION_ROOT` to the task's persistent personal-global directory,
`vlastudio-navigation-1015`, then use:

| Model | Directory below `${NAVIGATION_ROOT}/paper-reproduction` | `policy.pt` SHA256 |
|---|---|---|
| CMA | `cma-continuation-lr1e-6/epoch-4` | `30869101e833f6033ba8718f3bd76af613f0de2416788cac5f8ecd04abbc25ef` |
| Recurrent VLN-BERT | `vlnbert-continuation-lr1e-7/epoch-1` | `71380fc0bff6cc0b656933c626b76ba5a6ed3dbea9c68489eafc8609b7db9bde` |

Models, annotations, cached features and original logs remain in personal-global
storage; large model/data files are not committed to Git. MP3D scene data remain
on the platform dataset mount, as authorized by the user.

- [`acceptance.json`](acceptance.json): exact criteria, selected hashes, training
  updates, full results and hashes of all 11 original episode journals.
- [`cma-audit.json`](cma-audit.json), [`vlnbert-audit.json`](vlnbert-audit.json):
  training/validation separation, parameter changes and selected checkpoint audit.
- `*-val-unseen/**/evaluation.json`: unrounded metrics; `episodes.jsonl.gz`:
  complete per-episode actions and metrics, not a sample. Gzip is deterministic.
- `*/training_manifest.json`, `*/training_result.json`, `*/losses.json.gz`:
  exact episode selection, arguments, updates and loss history.
- [`runtime.json`](runtime.json), `assets/`, `places365-features/provenance.json`:
  runtime, upstream source revisions and asset checksums.
- Resolved `*.yaml` and JSON copies replace the personal-global prefix with
  `${NAVIGATION_ROOT}` for portability. Recorded config SHA256 values refer to
  the original resolved remote files, **before** this path normalization. Original
  files remain beside the remote experiment logs. Decompress journals before
  comparing their hashes with `acceptance.json`.

Run instructions and the fixed metric protocol are in
[`examples/09_navigation`](../../../../examples/09_navigation/README.md) and
[`PROTOCOL.md`](PROTOCOL.md).

References: [authors' corrected CMA reference](https://github.com/jacobkrantz/VLN-CE/tree/729d141b2ee10628061ada74dd3a5b9f70faeba5#baseline-performance),
[Recurrent VLN-BERT paper, Table 1](https://arxiv.org/html/2011.13922v2),
[official VLN-BERT code](https://github.com/YicongHong/Recurrent-VLN-BERT/tree/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e).
