#!/bin/bash

TASKNAME=libero_object
OUTPUT=./ckpt/qdp_zscore_pipe2 # Notice a standard OUTPUT dir should include key words "lora" and "qwen2_vl" for better load model(e.g. /root/path/lora_qwen2_vla_factory_sorting)

deepspeed --master_port 29604 --num_gpus=1 --num_nodes=1 ./train.py \
    --model_name qwen2vl_dp \
    --deepspeed scripts/zero2.json \
    --use_reasoning False \
    --use_prev_subtask False \
    --lora_enable True \
    --action_normalize zscore \
    --state_normalize zscore \
    --use_quantization False \
    --action_dim 7 \
    --state_dim 7 \
    --flash_attn True \
    --chunk_size 16 \
    --load_pretrain_dit False \
    --policy_head_type unet_diffusion_policy \
    --image_size_primary "(256,256)" \
    --image_size_wrist "(256,256)" \
    --task_name $TASKNAME \
    --model_name_or_path /inspire/hdd/global_user/wangzheng-240308120196/models/Qwen2-VL-2B-Instruct \
    --version v0 \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT \
    --max_steps 20 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --non_lora_lr 2e-4 \
    --lora_module llm,merger,vit \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --policy_class unet_diffusion_policy \
    --concat token_cat \
    --report_to tensorboard \
    --logging_dir $OUTPUT/log | tee $OUTPUT/log.log
    # 
    # --resume_from_checkpoint True \

for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo $OUTPUT

#   --resume_from_checkpoint True 
