#!/bin/bash
ENV=robomimic
TASKNAME=NutAssemblySquare_Panda

MODELNAME=act
NORM=zscore
CHUNKSIZE=50
DATASET=/inspire/hdd/project/robot-action/public/data/robomimic/square/ph

CKPT=/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Framework/ckpt/act_robomimic_square_zscore

FPS=50
ROLLOUT=2
PARALLEL=1
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
    --image_size_primary "(84,84)" \
    --image_size_wrist "(84,84)" \
    --max_timesteps 400 \ 
