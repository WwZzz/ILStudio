#!/bin/bash

TASKNAME=insertion_top
MODELNAME=diffusion_policy
NORM=zscore
OUTPUT=ckpt/${MODELNAME}_${TASKNAME}_${NORM}

python ./train.py \
    --task_name $TASKNAME \
    --output_dir $OUTPUT  \
    --model_name $MODELNAME \
    --use_reasoning False \
    --use_prev_subtask True \
    --action_normalize $NORM \
    --state_normalize $NORM \
    --state_dim 14 \
    --action_dim 14 \
    --chunk_size 100 \
    --image_size_primary "(640,480)" \
    --image_size_wrist "(256,256)" \
    --max_steps 400000 \
    --per_device_train_batch_size 64 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 50 \
    --learning_rate 1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=500 \
    --weight_decay 1e-6 \
    --logging_steps 100 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --preload_data True \
    --report_to tensorboard \
    --resume_from_checkpoint True \
    --logging_dir $OUTPUT/log | tee $OUTPUT/log.log
