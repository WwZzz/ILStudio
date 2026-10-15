# Examples

Run all commands from the repository root. VLAStudio 0.2.2+ includes base dependencies.
If VLAStudio is already installed, reuse that environment. For a new checkout, install once:

```bash
python -m pip install -e .
```

| Example | Environment |
| --- | --- |
| 01 ACT / ALOHA | Existing base; run Python directly |
| 02 Remote inference | Both default processes can share base |
| 03 Custom policy | Existing base |
| 04 Multitask SmolVLA | SmolVLA + LIBERO; reuse the environment from 07 |
| 05 Pi0.5 | OpenPI environment; base plus Tabletop-Sim for evaluation |
| 06 Custom robot | Existing base; GUI modes need Tkinter and a display |
| 07 LIBERO-Plus | SmolVLA + LIBERO-Plus; setup reuses its cached environment |
| 08 FastWAM | FastWAM + LIBERO; setup reuses its cached environment |

Do not reinstall base for each example. Install additional components once, then
activate the prepared environment and run the Python files. Sources, datasets,
assets and model weights are prepared separately where needed. Cache defaults to
`~/.cache/vlastudio`; set `VLASTUDIO_CACHE` to change it.
