# 09 — Navigation

Run from the repository root on Linux with Python 3.10 and an NVIDIA GPU.
Choose a simulator; each folder has its own setup, configuration, training and evaluation.

| Simulator | Task / policy | Example |
|---|---|---|
| Habitat | Continuous VLN-CE / CMA | [habitat/](habitat/README.md) |
| MatterSim | Viewpoint R2R / Recurrent VLN-BERT | [mattersim/](mattersim/README.md) |

Edit the paths in the selected folder's `config/`, then follow its README.
Both learned examples continue training from released weights.

## What is the navigation data?

Each task combines a **natural-language route instruction**, a scene, the agent's
observations and an expert action sequence. R2R annotations describe graph paths
through Matterport3D viewpoints; VLN-CE annotations provide continuous-simulator
episodes and expert primitive actions. Matterport3D scene assets are separate
from the instruction/action annotations and require authorized access.

For example, an illustrative instruction is “Leave the bedroom, turn left into
the hallway, and stop by the stairs.” The language is an input; the annotated
expert path/actions are training targets, not hints exposed during evaluation.
There is no supplied numeric local destination point in these two VLN examples.

| Example | A dataset item | Main visual input | Supervision |
|---|---|---|---|
| [Habitat / CMA](habitat/README.md#data-composition) | One instruction and its full expert sequence | Raw replayed RGB/depth | STOP/FORWARD/LEFT/RIGHT and training-only distance measurements |
| [MatterSim / VLN-BERT](mattersim/README.md#data-composition) | One instruction and its graph-path sequence | Published panorama features and raw candidate geometry, or rendered RGB | Simulator action IDs, ending in STOP |

Both expose original text as `raw_lang`. Datasets own multi-frame loading,
temporal alignment and raw data caches, following the FastWAM sequence dataset.
RGB pixels and precomputed visual features both use `image`;
`reasoning.image_type` distinguishes their representations. Extra metadata,
including candidates, source provenance, frame alignment and auxiliary supervision,
lives in the reserved `reasoning` dictionary. Depth remains a separate modality.
Policy processors convert these samples into model inputs; the shared data task
cache can store their outputs. Candidate lists in MatterSim
describe reachable neighbors; they do not identify the correct next neighbor.
The policy chooses from them. Per-folder READMEs show exact sample layouts and
explain how training labels differ from live observations.

[Installation and interfaces](../../docs/navigation.md) ·
[Data and weights](../../docs/navigation_assets.md) ·
[Recorded results](../../docs/evidence/navigation/standard-interface/README.md)
