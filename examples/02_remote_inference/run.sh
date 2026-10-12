#!/usr/bin/env bash
# Optional ACT convenience wrapper. Install both model and simulator dependencies
# in the active environment first; for separate environments use the README.
set -euo pipefail
python examples/02_remote_inference/serve.py &
python examples/02_remote_inference/evaluate.py
