#!/usr/bin/env bash
# Optional wrapper for the two ordinary venvs prepared in README.md.
set -euo pipefail
source .venv-pi05/bin/activate
python examples/05_finetune_vla_pi05/train.py
python examples/05_finetune_vla_pi05/serve.py &
source .venv-sim/bin/activate
export MUJOCO_GL=egl
python examples/05_finetune_vla_pi05/evaluate.py
