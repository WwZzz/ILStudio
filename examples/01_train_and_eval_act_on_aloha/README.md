# Train and evaluate ACT on ALOHA

Activate a Python 3.10 virtual environment and run from the repository root.
See the [environment guide](../../docs/environments.md) to create an environment
with pip or uv. Installation is separate from the workload; you can edit the
Python script or copy its API calls into your own `run.py`.

```bash
python -m pip install -e .
python examples/01_train_and_eval_act_on_aloha/train_and_evaluate.py
```

Prepare the demonstrations at `data/sim_transfer_cube_scripted`, or update
the task YAML's `dataset_path_list` to your own data path. The script trains ACT
and evaluates the resulting checkpoint on ALOHA-Sim.
