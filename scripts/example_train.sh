#!/bin/bash

TASKNAME=sim_transfer_cube_scripted # keep the same to the key in configuration.constants.TASK_CONFIG
MODELNAME=act # the policy name
NORM=zscore # the normalizer name, e.g., zscore, minmax, and
OUTPUT=ckpt/${MODELNAME}_${TASKNAME}_${NORM}_example # the checkpoint name that will be used for evaluation

# Most of the parameters are inherented from Huggingface-Trainer's args
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
    --max_steps 10000 \
    --per_device_train_batch_size 64 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 50 \
    --learning_rate 1e-4 \
    --lr_scheduler_type="constant" \
    --warmup_steps=0 \
    --warmup_ratio=0.0 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --preload_data True \
    --report_to tensorboard \
    --resume_from_checkpoint True \
    --logging_dir $OUTPUT/log | tee $OUTPUT/log.log


#   --resume_from_checkpoint True
