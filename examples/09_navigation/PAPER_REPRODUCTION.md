# VLN model acceptance protocol

Protocol fixed on 2026-10-15, before either complete validation result is available.

Two separate benchmarks are required. Neither expert trajectories, PointNav,
single-scene smoke tests, nor partial validation journals qualify as acceptance.

| Environment | Model | Complete validation split | Primary reference |
|---|---|---|---|
| Habitat / VLN-CE | CMA_PM_DA_Aug | R2R_VLNCE_v1-3_preprocessed val_unseen, 1,839 instruction episodes | Original paper SPL 0.30; authors' later official reevaluation SR 0.29, SPL 0.27, NE 7.60 m |
| MatterSim / R2R | Recurrent VLN-BERT, PREVALENT initialization | R2R val_unseen, 2,349 instruction episodes | Paper Table 1: SR 0.63, SPL 0.57, NE 3.93 m, TL 12.01 m |

Report SR, SPL, navigation error, trajectory length, episode count, scene count,
checkpoint hash, source revision, simulator/runtime version, and the complete
episode journal. Use greedy single-run inference, without search or exploration.
Report exact differences from the references; do not replace the paper target
with a lower number after observing results. Formal acceptance remains pending
until both full benchmarks meet the agreed references. On 2026-10-15 the user
explicitly selected the authors' corrected CMA reference (SR 0.29, SPL 0.27)
for acceptance. The original paper SPL 0.30 remains visible in every comparison.
R2R paper SR/SPL are published as integer percentages; retain unrounded values
in the evidence and show the comparison at the paper's published precision.

Habitat uses 0.25 m forward actions, 15 degree turns, 500 steps, RGB 224x224 and
normalized depth 256x256 (0–10 m), 90 degree horizontal FOV, and sensor height
1.25 m. Success requires an explicit STOP within 3 m. Original models were trained
with Habitat 0.1.7; the initial framework port uses 0.3.3, a material difference
that must be reported. The authors themselves document the 0.30 to 0.27 SPL
hardware/build discrepancy.

MatterSim follows the paper's panoramic candidate interface: 36 views at 30 degree
increments, official Places365 ResNet-152 features from the **current** viewpoint,
128-dimensional angle features, BERT uncased tokens (maximum length 80), and at
most 15 navigation decisions. STOP has no visual feature. Candidate actions must
turn toward the selected source view before moving, as in the authors' code.
No images from an unvisited candidate viewpoint may be used.

First evaluate an immutable official checkpoint to validate the implementation.
Then perform actual learning on training episodes only, saving the training
configuration, optimizer updates, loss history, parameter changes, and a separate
checkpoint. Evaluate the resulting checkpoint on the same complete validation
split. Any continuation or fine-tuning is labeled as such; it is not described as
a from-scratch reproduction of the paper's full training budget. Keep all training
labels, goals, shortest paths, and progress supervision out of policy inference.
Do not select subsets or terminate evaluation based on observed success rates.

Sources:

- [Official VLN-CE baseline configs](https://github.com/jacobkrantz/VLN-CE/tree/729d141b2ee10628061ada74dd3a5b9f70faeba5/vlnce_baselines/config/r2r_baselines)
- [Authors' corrected CMA reference](https://github.com/jacobkrantz/VLN-CE/tree/729d141b2ee10628061ada74dd3a5b9f70faeba5#baseline-performance)
- [Recurrent VLN-BERT paper, Table 1](https://arxiv.org/html/2011.13922v2)
- [Official VLN-BERT implementation](https://github.com/YicongHong/Recurrent-VLN-BERT/tree/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e)
