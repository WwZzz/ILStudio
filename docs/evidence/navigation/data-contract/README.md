# Navigation temporal data contract verification

This archive predates the subsequent [standard image/reasoning interface](../reasoning-interface/README.md).
Its original sample-layout receipts remain unchanged; the linked checks cover
the current layout and preserve the same short-run weight hashes.

The navigation datasets load temporal samples themselves, following FastWAM's
`HDF5SequenceDataset`: full episodes or fixed windows, aligned actions/states,
observation offsets and boundary padding masks. Public samples contain original
text and raw RGB/depth or published 2048-D panorama features. Policy processors
tokenize text and construct model inputs; they do not group episodes or manage
persistent caches. CMA's frozen encoder outputs use the existing data task cache.

The checks used the existing real Habitat/MatterSim installation and the saved
`17DRP5sb8fy` MP3D scene. They replayed official training demonstrations, including
Habitat RGB/depth, MatterSim feature mode and MatterSim RGB rendering. Fixed
windows reused the raw replay cache without opening a simulator and preserved
episode boundaries. [proof.json](proof.json) records sample fields and shapes.

Both policies then completed **two optimizer updates through `vla.train`**,
starting from official weights, with changed parameters and identical tensors
after strict save/reload. Saved checkpoints also passed offline inference with
original `raw_lang`; CMA's text encoding and VLN-BERT's dataset-supplied features
matched their source inputs. CMA's tokenization matched all **13,436** official
train/val_seen/val_unseen annotations with zero mismatches.

Focused CPU regressions: **139 passed, 2 skipped** (optional HTTP dependencies).
Coverage includes temporal padding/alignment, no cross-episode windows, stable
candidate action IDs, cache reader isolation, encoder cache invalidation and
empty-state shared-memory transport, alongside driving regressions.

These are interface and short-training checks. They do not repeat the full
navigation benchmarks or establish new SR/SPL estimates. Earlier benchmark
receipts remain in [standard-interface](../standard-interface/README.md).
