# Installation
```
cd benchmark/metaworld
uv sync
uv pip install metaworld
```

# Trouble Shooting
- `mujoco.FatalError: gladLoadGL error`: running the command in the shell `export MUJOCO_GL=egl`