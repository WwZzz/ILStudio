#!/usr/bin/env bash
# Optional wrapper for the prepared environments in README.md.
set -euo pipefail
policy_python="${POLICY_PYTHON:-$(vlastudio env path pi05)}"
sim_python="${SIM_PYTHON:-python}"
"$policy_python" examples/05_finetune_vla_pi05/train.py
"$policy_python" examples/05_finetune_vla_pi05/serve.py &
export MUJOCO_GL=egl
"$sim_python" examples/05_finetune_vla_pi05/evaluate.py
