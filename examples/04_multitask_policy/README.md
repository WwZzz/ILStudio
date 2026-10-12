# Multitask SmolVLA on LIBERO

Activate a Python 3.10 virtual environment and run from the repository root.
See the [environment guide](../../docs/environments.md) to create an environment
with pip or uv. Installation is separate from the workload; you can edit the
Python script or copy its API calls into your own `run.py`.

```bash
python -m pip install -e .
python examples/install_dependencies.py --policy smolvla --env libero.object
python examples/04_multitask_policy/train_and_evaluate.py
```

The script loads `libero.libero_object`, trains the policy in this example's
`smolvla.yaml`, and evaluates all ten `libero.object` tasks. Prepare LIBERO's
assets and point the dataset YAML at your local demonstrations before running.
For separate model/simulator environments, follow [Example 02](../02_remote_inference).
