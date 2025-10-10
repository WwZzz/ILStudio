# Installation
```shell
cd policy/octo/octo_pytorch
cp ../pyproject.toml pyproject.toml
uv sync # if uv was not found, use 'pip install uv' to install it
source .venv/bin/activate
uv pip install 'git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972'
uv pip install -e .
cd ../../..
```
