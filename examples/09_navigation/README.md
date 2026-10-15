# 09 — Navigation

Run from the repository root on Linux with Python 3.10 and an NVIDIA GPU.
Choose a simulator; each folder has its own setup, configuration, training and evaluation.

| Simulator | Task / policy | Example |
|---|---|---|
| Habitat | Continuous VLN-CE / CMA | [habitat/](habitat/README.md) |
| MatterSim | Viewpoint R2R / Recurrent VLN-BERT | [mattersim/](mattersim/README.md) |

Edit the paths in the selected folder's `config/`, then follow its README.
Both learned examples continue training from released weights.

[Installation and interfaces](../../docs/navigation.md) ·
[Data and weights](../../docs/navigation_assets.md) ·
[Recorded results](../../docs/evidence/navigation/paper/README.md)
