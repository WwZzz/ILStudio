#!/bin/bash

TASKNAME=sii_pick_twice
MODELNAME=act
NORM=zscore
OUTPUT=ckpt/${MODELNAME}_${TASKNAME}_${NORM} # Notice a standard OUTPUT dir should include key words "lora" and "qwen2_vl" for better load model(e.g. /root/path/lora_qwen2_vla_factory_sorting)

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
    --chunk_size 32 \
    --image_size_primary "(640,480)" \
    --image_size_wrist "(640,480)" \
    --max_steps 50000 \
    --per_device_train_batch_size 32 \
    --save_strategy steps \
    --save_steps 10000 \
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
