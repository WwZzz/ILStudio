# LoRA-finetune Pi0.5 on Tabletop-Sim

Run from the repository root with VLAStudio 0.2.2+ installed.
Use a separate Python 3.11 OpenPI environment; reuse base for the simulator.
Skip installation commands for components you have already prepared.

## Train and serve

```bash
vlastudio env create -n pi05 --policy openpi
vlastudio env activate pi05
python examples/05_finetune_vla_pi05/train.py
python examples/05_finetune_vla_pi05/serve.py
```

The installer applies OpenPI's required Transformers patch. Provide converted
`pi05_base_torch` weights at the path in `pi05.yaml` and prepare the
`jellyho/aloha_dish_drainer` dataset. See the [OpenPI guide](../../src/vlastudio/policy/openpi/README.md).
`exit` leaves the activated child shell.

## Evaluate in another terminal

Activate your existing Python 3.10 base environment. Add Tabletop-Sim once:

```bash
vlastudio env install --env tabletop_sim
python -m pip install --no-deps -e /path/to/Tabletop-Sim
```

Prepare its [source, submodules and assets](../../src/vlastudio/benchmark/tabletop_sim/README.md).
This uses the simulator's MuJoCo 3.2.0 and dm_control 1.0.21 pins.
If you need to preserve another simulator's versions, use a separate environment.
With the policy server listening:

```bash
MUJOCO_GL=egl python examples/05_finetune_vla_pi05/evaluate.py
```

Use a fresh output directory for each evaluation. OpenPI requires Linux.
