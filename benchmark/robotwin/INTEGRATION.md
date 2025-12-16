# RoboTwin Integration with ILStudio

## Integration Status: ✅ COMPLETE

RoboTwin has been successfully integrated into ILStudio following the project rules.

---

## Overview

RoboTwin is a dual-arm manipulation benchmark with 50 diverse tasks. This integration allows you to:
- Evaluate policies on RoboTwin tasks using ILStudio
- Use ILStudio's training pipeline with RoboTwin data
- Leverage RoboTwin's domain randomization and rich observations

---

## Key Features

### Environment Capabilities
- ✅ Dual-arm coordinated manipulation
- ✅ 50 diverse manipulation tasks
- ✅ Multiple observation modalities (RGB, depth, point cloud, segmentation)
- ✅ Domain randomization (background, lighting, table height)
- ✅ Both joint space (qpos) and end-effector (ee) control

### ILStudio Integration
- ✅ Follows `env-rule.mdc` guidelines
- ✅ Implements `create_env()` function
- ✅ `RoboTwinEnv` inherits from `MetaEnv`
- ✅ Implements `obs2meta()` and `meta2act()`
- ✅ Returns `MetaObs` format observations
- ✅ Supports `MetaAction` interface
- ✅ Compatible with `eval_sim.py`

---

## File Structure

```
benchmark/robotwin/
├── __init__.py           # Main integration (RoboTwinEnv class)
├── README.md             # Usage documentation
├── INTEGRATION.md        # This file
├── pyproject.toml        # Dependencies
├── test_integration.py   # Integration test script
└── RoboTwin/             # Original RoboTwin repository
    ├── envs/             # Task environments
    ├── task_config/      # Task configurations
    └── ...
```

---

## Installation

```bash
cd benchmark/robotwin/RoboTwin

# Install dependencies (creates .venv)
bash script/_install.sh

# Download assets
bash script/_download_assets.sh
```

---

## Configuration Format

```yaml
type: benchmark.robotwin.RoboTwinEnv
name: robotwin_<task>
args:
  task: <task_name>           # Python class name (e.g., 'pick_diverse_bottles')
  task_config: demo_clean     # 'demo_clean' or 'demo_randomized'
  max_timesteps: 1000         # Maximum steps per episode (required!)
  ctrl_space: qpos            # 'qpos' or 'ee'
  ctrl_type: abs              # Only 'abs' supported
  seed: 0
  camera_names:
    - head_camera
    - left_wrist_camera
    - right_wrist_camera
  image_size: [480, 640]      # [H, W]
```

---

## Observation Format (MetaObs)

```python
MetaObs(
    state=np.array(...),      # Robot state (qpos or ee based on ctrl_space)
    state_joint=np.array(...),# Joint positions (if available)
    image=np.array(...),      # (N, 3, H, W) RGB images from cameras
    raw_lang=str              # Language instruction (if set)
)
```

### State Dimensions
- **qpos mode**: `[left_joints(7), left_gripper(1), right_joints(7), right_gripper(1)]` = 16D
- **ee mode**: `[left_pose(7), left_gripper(1), right_pose(7), right_gripper(1)]` = 16D

---

## Action Format

### Input (MetaAction)
```python
MetaAction(
    action=np.array([...]),  # Action array
    ctrl_space='qpos',       # 'qpos' or 'ee'
    ctrl_type='abs'          # Only 'abs' supported
)
```

### Action Dimensions
- **qpos mode**: Same as state (16D for ALOHA)
- **ee mode**: Same as state (16D)

---

## Usage

### With eval_sim.py

⚠️ **MUST use RoboTwin's Python environment!**

```bash
# Correct - using RoboTwin's venv
benchmark/robotwin/.venv/bin/python eval_sim.py \
  -e robotwin_pick_bottles \
  -m <model_path> \
  --batch_size 0

# Wrong - using system/ILStudio python
python eval_sim.py -e robotwin_pick_bottles -m <model>
```

### Sequential Mode Required

⚠️ **Always use `--batch_size 0`** (Sapien limitation)

```bash
# ✅ Correct
--batch_size 0

# ❌ Wrong - will fail
--batch_size 4
```

---

## Available Tasks

50 tasks in `RoboTwin/envs/`:

### Pick & Place
- `pick_diverse_bottles`
- `place_bread_basket`
- `place_bread_skillet`
- `place_burger_fries`
- `place_can_basket`
- `place_cans_plasticbox`
- `place_container_plate`
- `place_empty_cup`
- `place_fan`
- `place_mouse_pad`
- `place_object_basket`
- `place_object_scale`
- `place_object_stand`
- `place_phone_stand`
- `place_shoe`
- `place_dual_shoes`
- `place_a2b_left`
- `place_a2b_right`

### Stacking
- `stack_blocks_two`
- `stack_blocks_three`
- `stack_bowls_two`
- `stack_bowls_three`
- `blocks_ranking_rgb`
- `blocks_ranking_size`

### Opening/Closing
- `open_laptop`
- `open_microwave`

### Manipulation
- `press_stapler`
- `shake_bottle`
- `shake_bottle_horizontally`
- `rotate_qrcode`
- `scan_object`
- `click_alarmclock`
- `click_bell`
- `beat_block_hammer`
- `adjust_bottle`
- `turn_switch`
- `stamp_seal`
- `pick_dual_bottles`

### Object Interaction
- `grab_roller`
- `handover_block`
- `handover_mic`
- `hanging_mug`
- `lift_pot`
- `move_can_pot`
- `move_pillbottle_pad`
- `move_playingcard_away`
- `move_stapler_pad`
- `dump_bin_bigbin`
- `put_bottles_dustbin`
- `put_object_cabinet`

---

## Testing

### Quick Integration Test
```bash
benchmark/robotwin/.venv/bin/python benchmark/robotwin/test_integration.py
```

### With Dummy Policy
```bash
benchmark/robotwin/.venv/bin/python eval_sim.py \
  -m __dummy-16random \
  -e robotwin_pick_bottles \
  --batch_size 0 \
  --num_rollout 1
```

---

## Important Notes

### Environment-Specific Python Required
RoboTwin has complex dependencies (Sapien, PyTorch3D, cuRobo) that require its own virtual environment. Always use `benchmark/robotwin/.venv/bin/python`.

### Sequential Mode Only
Sapien (like CoppeliaSim) cannot be safely forked in multiple processes. Always use `--batch_size 0`.

### Episode Initialization
Some episodes may fail to initialize due to unstable object placement. This is expected behavior in RoboTwin. The environment will automatically retry with a different seed.

### TOPP Planning
RoboTwin uses TOPP (Time-Optimal Path Parameterization) for smooth trajectory planning. This adds computation time but ensures safe, collision-free motion.

---

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'sapien'`

**Solution**: Use RoboTwin's Python
```bash
benchmark/robotwin/.venv/bin/python eval_sim.py ...
```

### Episode Initialization Fails
**Problem**: "Objects is unstable" or setup fails

**Solution**: This is normal. The environment automatically tries new seeds. If it fails repeatedly, the task may be too constrained.

### Slow Performance
**Problem**: Episodes run slowly

**Solution**: 
- Disable rendering: `render_freq: 0` (default)
- Use qpos control instead of ee (faster, no motion planning)
- Reduce `max_timesteps` if appropriate

### Action Dimension Mismatch
**Problem**: `ValueError: Action dimension mismatch`

**Solution**: Check `get_action_dim()`. For ALOHA-Agilex dual-arm: 16D (7+1 per arm)

---

## Performance Considerations

- **Computation**: Dual-arm planning is compute-intensive
- **Episode Time**: 30-60 seconds per episode (varies by task)
- **Memory**: ~2GB per environment instance
- **GPU**: Not required but helpful for rendering/point clouds

---

## Comparison with Other Envs

| Feature | RoboTwin | RLBench | CALVIN | LIBERO |
|---------|----------|---------|--------|--------|
| Arms | Dual | Single | Single | Single |
| Tasks | 50 | 100+ | 34 | 130 |
| Control | qpos/ee | joint/ee | ee | ee |
| Parallel | ❌ | ❌ | ✅ | ✅ |
| Domain Rand | ✅ | ❌ | ❌ | ✅ |

---

## Integration Checklist

- [x] `create_env()` function implemented
- [x] `RoboTwinEnv` inherits from `MetaEnv`
- [x] `obs2meta()` converts to MetaObs
- [x] `meta2act()` converts from MetaAction
- [x] `reset()` returns MetaObs
- [x] `step()` executes actions
- [x] `close()` cleanups environment
- [x] `get_action_dim()` returns correct dimension
- [x] Config file created
- [x] README with installation
- [x] pyproject.toml with dependencies
- [x] Test script created
- [x] Compatible with eval_sim.py
- [x] Handles max_timesteps correctly
- [x] Follows env-rule.mdc guidelines

---

## Status: ✅ READY FOR USE

RoboTwin is fully integrated and ready for:
- Policy evaluation
- Data collection
- Training with ILStudio pipeline
- Custom task development

**Remember**: Use `benchmark/robotwin/.venv/bin/python` and `--batch_size 0`!

