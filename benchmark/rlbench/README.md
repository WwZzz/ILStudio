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

## Available Tasks

Create config files for other RLBench tasks:
- ReachTarget, PickAndLift, PickUpCup
- OpenDrawer, CloseJar, PushButton
- StackBlocks, SlideBlockToTarget
- And 100+ more tasks...

See full list: https://github.com/stepjam/RLBench/tree/master/rlbench/tasks

