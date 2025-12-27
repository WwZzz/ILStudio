# RLBench Environment

## Installation
```shell
# Download and install CoppeliaSim
# set env variables
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

uv pip install git+https://github.com/stepjam/RLBench.git
```



## Usage

⚠️ **IMPORTANT**: RLBench does **NOT** support parallel execution because CoppeliaSim cannot run in multiple processes.

Always use `--batch_size 0` to force sequential execution:

```bash
# Evaluate on ReachTarget task
python eval_sim.py -e rlbench_reach -m <model_path> --batch_size 0

# Evaluate on PickAndLift task  
python eval_sim.py -e rlbench_pick -m <model_path> --batch_size 0

# Multiple rollouts in sequence
python eval_sim.py -e rlbench_reach -m <model_path> --batch_size 0 --num_rollout 10
```

## Dataset Generation

Generate HDF5 datasets for training using the built-in demo collection:

```bash
# Generate 50 demos for ReachTarget task
python benchmark/rlbench/generate_dataset.py \
    --env_config configs/env/rlbench_reach.yaml \
    --output_dir data/rlbench/reach_target \
    --num_demos 50

# Generate demos for other tasks (create corresponding env config first)
python benchmark/rlbench/generate_dataset.py \
    --env_config configs/env/rlbench_pick.yaml \
    --output_dir data/rlbench/pick_and_lift \
    --num_demos 100
```

### Batch Generation

Generate datasets for all RLBench tasks automatically:

```bash
# Generate datasets for all tasks (50 demos per task, default)
bash benchmark/rlbench/generate_all_datasets.sh

# Generate with custom number of demos
bash benchmark/rlbench/generate_all_datasets.sh --num_demos 100

# Custom output directory
bash benchmark/rlbench/generate_all_datasets.sh --base_dir /path/to/output --num_demos 50

# Disable headless mode (for debugging)
bash benchmark/rlbench/generate_all_datasets.sh --no-headless --num_demos 10
```

**Note**: The script automatically:
- Finds all `rlbench_*.yaml` config files (excluding `*_ee.yaml` versions)
- Skips tasks if dataset already exists (with confirmation prompt)
- Generates one dataset per task (joint and ee versions share the same dataset)
- Provides progress summary at the end

### Generated Data Format

Each episode is saved as an HDF5 file with the following structure:
- `/action`: Action sequence (T, 8) - depends on ctrl_space
- `/state`: State sequence (T, 8) - depends on ctrl_space
- `/observations/joint_positions`: (T, 7)
- `/observations/joint_velocities`: (T, 7)
- `/observations/gripper_pose`: (T, 7)
- `/observations/gripper_open`: (T, 1)
- `/observations/images/front`: (T, H, W, 3)
- `/observations/images/wrist`: (T, H, W, 3)
- `/language_instruction`: Task descriptions
- `/episode_len`: Episode length

### Training with Generated Data

```bash
# Train ACT policy
python train.py --policy act --task rlbench_reach --output_dir ckpt/act_rlbench_reach

# Train Diffusion Policy
python train.py --policy diffusion_policy --task rlbench_reach --output_dir ckpt/dp_rlbench_reach
```

## Available Tasks

Create config files for other RLBench tasks:
- ReachTarget, PickAndLift, PickUpCup
- OpenDrawer, CloseJar, PushButton
- StackBlocks, SlideBlockToTarget
- And 100+ more tasks...

See full list: https://github.com/stepjam/RLBench/tree/master/rlbench/tasks

# TroubleShooting 

## Headless Server 
```shell
nohup X :99 & disown
export DISPLAY=:99
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```
