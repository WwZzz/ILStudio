# Train and evaluate ACT on ALOHA

Activate a Python 3.10 virtual environment and run from the repository root.
See the [environment guide](../../docs/environments.md) to create an environment
with pip or uv. Installation is separate from the workload; you can edit the
Python script or copy its API calls into your own `run.py`.

```bash
python -m pip install -e ".[base]"
python examples/01_train_and_eval_act_on_aloha/train_and_evaluate.py
```

The script uses `training_config.yaml` for ACT and requires all 50 transfer-cube
demonstrations before training. It saves a fresh checkpoint under
`checkpoints/act_aloha_full` and evaluates it on ALOHA-Sim.
