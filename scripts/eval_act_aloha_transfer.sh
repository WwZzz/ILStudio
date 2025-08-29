#!/bin/bash

export MUJOCO_GL=egl # enable this line for headless ubuntu server 

# benchmark infomation
ENV=aloha
TASKNAME=transfer_cube
CAM_IDS="[0]"
IMAGE_SIZE="(640,480)"

# policy information
MODELNAME=act
CHUNKSIZE=100
CKPT=/inspire/hdd/project/robot-action/wangzheng-240308120196/IL-Studio/ckpt/act_transfer_human_zscore

# evaluation information
FPS=50
ROLLOUT=50
PARALLEL=5
MAX_STEPS=400
OUTPUT=results/${MODELNAME}_${TASKNAME}_human

python eval.py --env_name $ENV \
    --task $TASKNAME \
    --model_name $MODELNAME \
    --model_name_or_path $CKPT \
    --chunk_size $CHUNKSIZE \
    --save_dir $OUTPUT \
    --num_rollout $ROLLOUT \
    --num_envs $PARALLEL \
    --fps $FPS \
    --camera_ids $CAM_IDS \
    --image_size_primary $IMAGE_SIZE \
    --image_size_wrist $IMAGE_SIZE \
    --max_timesteps $MAX_STEPS \
    # --use_spawn True \