# LoRA-finetune Pi0.5 on Tabletop-Sim

This example trains with `vla.train`, serves with `vla.serve`, and evaluates
with `Environment.evaluate`. Run from the repository root. The model uses
Python 3.11; the simulator uses Python 3.10, so prepare two environments.

## Train and serve

```bash
python -m pip install uv
python -m uv venv --python 3.11 --seed .venv-pi05
source .venv-pi05/bin/activate
python -m pip install uv
python -m uv pip install -e . -r requirements/openpi.txt --override requirements/openpi-overrides.txt
cp -r .venv-pi05/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/. .venv-pi05/lib/python3.11/site-packages/transformers/
python examples/05_finetune_vla_pi05/train.py
python examples/05_finetune_vla_pi05/serve.py
```

Provide the converted `pi05_base_torch` weights at the path in `pi05.yaml`,
and prepare the `jellyho/aloha_dish_drainer` LeRobot dataset before training.
The shell copy above applies OpenPI's required Transformers patch; the library does not install or patch packages at runtime.
See the [OpenPI guide](../../src/vlastudio/policy/openpi/README.md) for weight conversion.

## Evaluate in another terminal

```bash
python -m uv venv --python 3.10 --seed .venv-sim
source .venv-sim/bin/activate
python -m pip install -e . -r requirements/tabletop_sim.txt
```

Install the [Tabletop-Sim source and assets](../../src/vlastudio/benchmark/tabletop_sim/README.md)
in this simulator environment, including its Git submodules. Install that source with `python -m pip install --no-deps -e /path/to/Tabletop-Sim`. The requirements file
already supplies MuJoCo 3.2.0 and dm_control 1.0.21. Then run:

```bash
python examples/05_finetune_vla_pi05/evaluate.py
```

For Windows, activate these environments with `.venv-pi05\Scripts\Activate.ps1`
and `.venv-sim\Scripts\Activate.ps1`. OpenPI's Linux/glibc requirements still apply.
Keep the server running and wait until it is listening before evaluating.
On a headless GPU server, set `os.environ["MUJOCO_GL"] = "egl"` near the start
of `evaluate.py`. Use a fresh output directory for each evaluation.
