#!/bin/bash

export DISPLAY=:0          # 根据 echo $DISPLAY 调整

ENV=aloha
TASKNAME=transfer_cube_top

MODELNAME=act
NORM=zscore
CHUNKSIZE=50
# DATASET=/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_insertion_scripted
DATASET=/home/agilex/wz/data/aloha_sim/sim_transfer_cube_scripted
CKPT=/home/agilex/wz/code/IL-Studio/ckpt/act_agilex_transfer_cube_zscore_long/checkpoint-40

FPS=50
ROLLOUT=50
PARALLEL=2
OUTPUT=results/${MODELNAME}_${TASKNAME}_${NORM}

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
    --use_spawn True \
