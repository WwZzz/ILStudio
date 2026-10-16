# Standard image and reasoning interface

Navigation samples now use the reserved `reasoning` dictionary for candidate
metadata, feature provenance, observation masks, frame IDs and auxiliary targets.
The top-level `image` contains either RGB pixels or float32 `[T,36,2048]` panorama
features, distinguished by `reasoning.image_type` (`rgb` / `features`). Depth
remains a separate modality. Bench2Drive uses the same extension boundary for
frame provenance and auxiliary training supervision.

[proof.json](proof.json) records real Habitat RGBD and MatterSim RGB/feature
replay, two optimizer steps per model through the standard training API, strict
checkpoint reload, offline inference and a live environment step through
`MetaPolicy`. Both trained weight-file hashes equal those from the preceding
[data contract checks](../data-contract/proof.json); moving fields did not change
the resulting short-run weights. Original benchmark scores are not rerun here.

The focused regression suite passes **154 tests**, including shared-memory and
HTTP batching/serialization. Feature tensors retain their dtype and exact values,
including negative values and values above 255, through pixel transforms, legacy
resize preparation and JPEG-configured HTTP transport. TCP batches from nested
supervision match the previous flat representation exactly. Legacy navigation
clients remain readable; new environment producers use `reasoning.navigation`.
