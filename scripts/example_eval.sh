#!/bin/bash
export MUJOCO_GL=glfw # egl for headless ubuntu server
# export DM_CONTROL_RENDERER=software
export NUMPY_EXPERIMENTAL_DTYPE_API=1

ENV=aloha
TASKNAME=transfer_cube_top

# Optional: specify dataset_id if you have multiple datasets in normalize.json
# Leave empty to use the first dataset (default)
DATASET_ID=""
CKPT=ckpt/act_sim_transfer_cube_scripted_zscore_example
OUTPUT=results/act_${TASKNAME}_example

FPS=50
ROLLOUT=10
PARALLEL=5

# Task-related parameters (action_dim, state_dim, chunk_size, camera_names, etc.) are now loaded from task config
python eval.py --env_name $ENV \
    --task $TASKNAME \
    --model_name_or_path $CKPT \
    --save_dir $OUTPUT \
    --num_rollout $ROLLOUT \
    --num_envs $PARALLEL \
    --dataset_id $DATASET_ID \
    --fps $FPS \
    --camera_ids "[0]" \
    --max_timesteps 400 \
    --use_spawn
