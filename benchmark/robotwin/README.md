# RoboTwin Environment for ILStudio

https://github.com/TianxingChen/RoboTwin

RoboTwin is a dual-arm manipulation benchmark with 50 diverse tasks, supporting domain randomization and various observation modalities.

## Installation

⚠️ **IMPORTANT**: RoboTwin requires its own virtual environment.

```bash
cd benchmark/robotwin/RoboTwin

# Install dependencies (creates .venv)
bash script/_install.sh

# Download assets
bash script/_download_assets.sh
```

## Features

- 🤖 **Dual-arm manipulation**: Coordinated bimanual control
- 🎯 **50 diverse tasks**: Pick, place, stack, open, close, etc.
- 📷 **Rich observations**: RGB, depth, point cloud, segmentation
- 🎲 **Domain randomization**: Background, lighting, table height
- 🔧 **Flexible control**: Joint space (qpos) or end-effector (ee) control

## Usage with ILStudio

RoboTwin must run with its own Python environment:

```bash
# Use RoboTwin's Python
benchmark/robotwin/.venv/bin/python eval_sim.py -e robotwin_pick_bottles -m <model> --batch_size 0
```

⚠️ **Sequential mode only** (`--batch_size 0`) due to Sapien limitations.

## Configuration

Create task configs in `configs/env/robotwin_<task>.yaml`:

```yaml
type: benchmark.robotwin.RoboTwinEnv
name: robotwin_<task>
args:
  task: pick_diverse_bottles  # Task class name
  task_config: demo_clean     # Config file (demo_clean or demo_randomized)
  max_timesteps: 1000         # Maximum steps per episode
  ctrl_space: qpos            # 'qpos' or 'ee'
  ctrl_type: abs              # Only 'abs' supported
  seed: 0
  camera_names:
    - head_camera
    - left_wrist_camera
    - right_wrist_camera
  image_size: [480, 640]
  # Optional: specify custom RoboTwin path if not in default location
  # robotwin_root: /path/to/RoboTwin
```

### Flexible Path Configuration

If your RoboTwin installation is in a custom location, specify it in the config:

```yaml
args:
  robotwin_root: /custom/path/to/RoboTwin
  # ... other args
```

This resolves relative paths (like `./assets/objects/objaverse/list.json`) correctly regardless of where you run ILStudio from.

## Available Tasks

50 tasks available in `RoboTwin/envs/`:
- **Pick & Place**: `pick_diverse_bottles`, `place_bread_basket`, etc.
- **Stacking**: `stack_blocks_two`, `stack_bowls_three`, etc.
- **Opening**: `open_laptop`, `open_microwave`, etc.
- **Manipulation**: `press_stapler`, `shake_bottle`, `rotate_qrcode`, etc.

## Testing

```bash
# Quick test with dummy policy
benchmark/robotwin/.venv/bin/python eval_sim.py \
  -m __dummy-16random \
  -e robotwin_pick_bottles \
  --batch_size 0 \
  --num_rollout 1
```

## Notes

- Uses Sapien physics engine (similar to ManiSkill2)
- **No planner mode**: Motion planning disabled for policy evaluation (faster, no GPU required)
- Some episodes may fail initialization (unstable object placement)
- Success detection: Environment returns `done=True` only when task succeeds
- Performance: ~1-2 steps/sec in headless mode with single camera

## Return Values (ILStudio Convention)

- `done`: Task success status (`True` only when task succeeds)
- `info['success']`: Explicit task success flag
- `info['terminated']`: Episode ended (success or timeout)
- `info['truncated']`: Episode timed out without success
