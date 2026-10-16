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
| [Habitat / CMA](habitat/README.md#data-composition) | One instruction and its full expert sequence | Replayed RGB/depth or cached encoder features | STOP/FORWARD/LEFT/RIGHT and training-only progress |
| [MatterSim / VLN-BERT](mattersim/README.md#data-composition) | One instruction and its graph-path sequence | Cached panorama features and candidate angles | Correct candidate at each step, then STOP |

Both expose original text as `raw_lang`. The example-specific caches retain the
tokens/features their released baselines require. Candidate lists in MatterSim
describe reachable neighbors; they do not identify the correct next neighbor.
The policy chooses from them. Per-folder READMEs show exact sample layouts and
explain how training labels differ from live observations.

[Installation and interfaces](../../docs/navigation.md) ·
[Data and weights](../../docs/navigation_assets.md) ·
[Recorded results](../../docs/evidence/navigation/standard-interface/README.md)
