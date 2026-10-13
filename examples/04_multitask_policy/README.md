# Multitask SmolVLA on LIBERO

Base does not include SmolVLA or LIBERO. Prepare their additions once:

```bash
vlastudio env create -n smolvla --policy smolvla --env libero
vlastudio env activate smolvla
python examples/04_multitask_policy/train_and_evaluate.py
```

Skip `env create` when that environment is ready. If you already prepared
[Example 07](../07_custom_sim_env_libero_plus), reuse its Python environment
instead of creating another one. Select standard LIBERO's sources and assets:

```bash
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/envs/example07/bin/activate"
python examples/07_custom_sim_env_libero_plus/setup.py --backend standard
source "${VLASTUDIO_CACHE:-$HOME/.cache/vlastudio}/libero-standard/env.sh"
python examples/04_multitask_policy/train_and_evaluate.py
```

The script loads `libero.libero_object`, trains the policy in this example's
`smolvla.yaml`, and evaluates all ten `libero.object` tasks. Prepare LIBERO's
assets and point the dataset YAML at your local demonstrations before running.
For separate model/simulator environments, follow [Example 02](../02_remote_inference).

LIBERO needs its source checkout and assets; set `LIBERO_ROOT` to that checkout. See [Example 07](../07_custom_sim_env_libero_plus) for source and asset preparation.
