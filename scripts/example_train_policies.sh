#!/bin/bash

# Example script showing how to train different policies using the new configuration system

TASKNAME=sim_transfer_cube_scripted
NORM=zscore
OUTPUT_BASE=ckpt

# Function to train a policy
train_policy() {
    local policy_config_path=$1
    local policy_name=$(basename "$policy_config_path" .yaml)
    local output_dir=${OUTPUT_BASE}/${policy_name}_${TASKNAME}_${NORM}_example
    
    echo "Training policy config: $policy_config_path"
    echo "Output directory: $output_dir"
    
    python ./train.py \
        --task_name $TASKNAME \
        --output_dir $output_dir \
        --policy_config $policy_config_path \
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
        --per_device_train_batch_size 8 \
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
        --dataloader_pin_memory False \
        --preload_data False \
        --report_to tensorboard \
        --resume_from_checkpoint True \
        --logging_dir $output_dir/log | tee $output_dir/log.log
}

# Train different policies
echo "Available policies:"
echo "1. ACT (Action Chunking with Transformers)"
echo "2. Qwen2DP (Qwen2 Vision-Language Diffusion Policy)"
echo "3. Qwen2.5DP (Qwen2.5 Vision-Language Diffusion Policy)"
echo "4. Diffusion Policy"
echo "5. DiVLA (Diffusion Vision-Language Action)"
echo "6. RDT (Robotic Decision Transformer)"
echo ""

# Uncomment the policy you want to train:

# ACT - Fast and efficient transformer-based policy
train_policy "configs/policy/act.yaml"

# Qwen2DP - Vision-language model with diffusion policy
# train_policy "configs/policy/qwen2dp.yaml"

# Qwen2.5DP - Updated Qwen2.5-VL model
# train_policy "configs/policy/qwen25dp.yaml"

# Diffusion Policy - Pure diffusion-based approach
# train_policy "configs/policy/diffusion_policy.yaml"

# DiVLA - Vision-language diffusion policy
# train_policy "configs/policy/divla.yaml"

# RDT - Robotic Decision Transformer
# train_policy "configs/policy/rdt.yaml"

echo "Training completed!"
