# DeepMind Control Suite (DMC) Benchmark

This module provides integration with DeepMind Control Suite for ILStudio, enabling training and evaluation of image-based reinforcement learning algorithms.

## Installation

```bash
# Install DMC
pip install dm_control

# Install dmc2gym wrapper
pip install dmc2gym
```

## Usage

### Create Environment

```python
from benchmark.dmc import create_env, make_dmc_env

# Using config object
class Config:
    task = 'cheetah_run'
    image_size = 84
    action_repeat = 2
    frame_stack = 3
    seed = 0

env = create_env(Config())

# Direct creation
env = make_dmc_env(
    env_name='walker_walk',
    seed=0,
    image_size=84,
    action_repeat=2,
    frame_stack=3,
)
```

### Available Tasks

```python
from benchmark.dmc import list_tasks, get_task_description

# List all tasks
print(list_tasks())

# Get task description
print(get_task_description('cheetah_run'))
```

## Features

### Frame Stacking

Stack multiple frames along the channel dimension to provide temporal information:

```python
# Original: (3, 84, 84) - single RGB frame
# With frame_stack=3: (9, 84, 84) - 3 stacked frames
```

### Action Repeat

Repeat each action multiple times to reduce the effective horizon:

```python
# With action_repeat=2, env.step(action) internally does:
# for _ in range(2):
#     obs, r, done, info = env.step(action)
#     total_reward += r
```

### Normalized Actions

Actions are automatically normalized to [-1, 1] range.

## Environment Wrappers

1. **FrameStack**: Stack k consecutive frames
2. **ActionRepeat**: Repeat actions for sample efficiency  
3. **NormalizeActions**: Normalize action space to [-1, 1]
4. **DMCObservationWrapper**: Convert to ILStudio's MetaObs format

## Configuration

See `configs/env/dmc.yaml`:

```yaml
type: benchmark.dmc
task: cartpole_swingup
image_size: 84
action_repeat: 2
frame_stack: 3
```

## Supported Tasks

| Category | Tasks |
|----------|-------|
| **Easy** | cartpole_balance, reacher_easy, point_mass_easy |
| **Medium** | cartpole_swingup, cheetah_run, walker_walk |
| **Hard** | humanoid_walk, quadruped_run, reacher_hard |





