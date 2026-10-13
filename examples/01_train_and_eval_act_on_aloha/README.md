# Train and evaluate ACT on ALOHA

Use your existing VLAStudio base environment and run from the repository root.
No additional packages or virtual environment are needed. See [setup](../README.md).

```bash
python examples/01_train_and_eval_act_on_aloha/train_and_evaluate.py
```

The script uses `training_config.yaml` for ACT and requires all 50 transfer-cube
demonstrations before training. It saves a fresh checkpoint under
`checkpoints/act_aloha_full` and evaluates it on ALOHA-Sim.
