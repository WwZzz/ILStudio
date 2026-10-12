# So101 Robot Implementation

This directory contains the implementation for So101 robotic arms, following the same pattern as the Koch robot implementation.

## Overview

The So101 implementation provides:
- **Follower Robot**: `So101FollowerWithCamera` - Controls the So101 follower arm with integrated camera support
- **Leader Teleoperator**: `So101Leader` - Captures movements from a So101 leader arm for teleoperation

## Installation
From the repository root, activate a dedicated Python 3.10 environment and install:

```bash
python -m pip install -e .
vlastudio env install --policy smolvla
python -m pip install "lerobot[feetech]==0.3.3"
```

The SmolVLA recipe supplies a compatible Torch/LeRobot runtime, including shared
data and deployment dependencies. The Feetech extra supplies the motor SDK.
For uv-managed setup, run `uv sync --python 3.10`, activate `.venv`, then run the
last two commands above. See the [environment guide](../../../../../docs/environments.md).

## Architecture

The implementation follows the standard robot interface pattern:

```
So101FollowerWithCamera (BaseRobot)
├── Camera integration (OpenCV)
├── Joint control (6 DOF)
├── Observation collection
└── Action execution

So101Leader (BaseTeleopDevice)  
├── Joint position reading
├── Action conversion
└── Communication management
```

## Configuration Files

### Robot Configurations
- `src/vlastudio/configs/robot/so101.yaml` - Linux configuration (`-r so101`)
- `src/vlastudio/configs/robot/so101_win.yaml` - Windows configuration (`-r so101_win`)

### Teleop Configurations  
- `src/vlastudio/configs/teleop/so101_leader.yaml` - Linux configuration (`-t so101_leader`)
- `src/vlastudio/configs/teleop/so101_leader_win.yaml` - Windows configuration (`-t so101_leader_win`)

Paths above are relative to the repository root. Copy the YAML files to your own
directory to customize them, then pass the file paths to `-r` and `-t`.

## Implementation Notes

### Current Status
This implementation uses the official lerobot support for So101 robots, providing a complete integration.

The implementation leverages:

1. **Official lerobot Support**: Uses `lerobot.robots.so101_follower.SO101Follower` and `lerobot.teleoperators.so101_leader.SO101Leader`
2. **Camera Integration**: Built-in camera support through lerobot's configuration system
3. **Standardized Interface**: Follows the same patterns as other robots in the codebase

### Configuration

Make sure to adjust the following in the config files:
1. **Communication Ports**: Set the correct COM ports for your hardware
2. **Camera Settings**: Configure camera indices and resolution
3. **Robot IDs**: Use appropriate identifiers for your setup

## Usage

### Training Data Collection
```bash
# Start both the leader and follower, recording HDF5 episodes
vlastudio collect --runtime current -r so101 -t so101_leader -o data/so101_pick_place --dataset-format hdf5
```

### Policy Evaluation
```bash
# Connect to an already running policy server
vlastudio infer --runtime current -r so101 -m localhost:5000

# Or load a checkpoint locally (install its model dependencies first)
vlastudio infer --runtime current -r so101 -m checkpoints/my_policy
```

## Hardware Requirements

- So101 Follower robot arm
- So101 Leader robot arm (for teleoperation)
- USB camera (for vision)
- Appropriate communication interfaces (USB, Serial, etc.)

## Troubleshooting

1. **Connection Issues**: Check COM port settings and hardware connections
2. **lerobot Installation**: Ensure lerobot is properly installed with So101 support
3. **Joint Mapping**: Verify motor names match your specific So101 configuration
4. **Permissions**: Ensure proper permissions for device access on Linux systems
