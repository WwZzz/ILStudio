#!/bin/bash
export MUJOCO_GL=egl

ENV=aloha
TASKNAME=transfer_cube_top
MODELNAME=act
CHUNKSIZE=100

DATASET=/path/to/sim_transfer_cube_scripted
CKPT=/path/to/checkpoint
OUTPUT=results/${MODELNAME}_${TASKNAME}_${CHUNKSIZE}_example

FPS=50
ROLLOUT=10
PARALLEL=5

python eval.py --env_name $ENV \
    --task $TASKNAME \
    --model_name $MODELNAME \
    --model_name_or_path $CKPT \
    --save_dir $OUTPUT \
    --num_rollout $ROLLOUT \
    --num_envs $PARALLEL \
    --dataset_dir $DATASET \
    --fps $FPS \
    --freq $CHUNKSIZE \
    --chunk_size $CHUNKSIZE \
    --camera_ids "[0]" \
    --image_size_primary "(640,480)" \
    --image_size_wrist "(256,256)" \
    --max_timesteps 400 \
