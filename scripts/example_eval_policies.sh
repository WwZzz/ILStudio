#!/bin/bash

# Example evaluation script using the new policy config system
# This script demonstrates how to evaluate different policies using path-based configuration

export MUJOCO_GL=glfw
export DM_CONTROL_RENDERER=software
export NUMPY_EXPERIMENTAL_DTYPE_API=1

TASKNAME=sim_transfer_cube_scripted
ENVNAME=aloha
NUM_ROLLOUT=4
NUM_ENVS=2
SAVE_DIR=results/eval_policies

echo "Policy Evaluation Examples"
echo "=========================="

# Function to evaluate a policy
eval_policy() {
    local policy_config_path=$1
    local policy_name=$(basename "$policy_config_path" .yaml)
    local output_dir=${SAVE_DIR}/${policy_name}_${TASKNAME}
    
    echo "Evaluating policy: $policy_config_path"
    echo "Output directory: $output_dir"
    
    python eval.py \
        --policy_config $policy_config_path \
        --env_name $ENVNAME \
        --task $TASKNAME \
        --num_rollout $NUM_ROLLOUT \
        --num_envs $NUM_ENVS \
        --save_dir $output_dir \
        --max_timesteps 400 \
        --fps 50 \
        --image_size "(640,480)" \
        --ctrl_space joint \
        --ctrl_type abs \
        --camera_ids "[0]"
}

# List of available policies (using full paths)
POLICIES=(
    "configs/policy/act.yaml"
    "configs/policy/qwen2dp.yaml" 
    "configs/policy/qwen25dp.yaml"
    "configs/policy/diffusion_policy.yaml"
    "configs/policy/divla.yaml"
    "configs/policy/rdt.yaml"
)

echo "Available policies:"
for policy in "${POLICIES[@]}"; do
    echo "  - $policy"
done
echo ""

# Uncomment the policy you want to evaluate:

# ACT - Fast and efficient transformer-based policy
eval_policy "configs/policy/act.yaml"

# Qwen2DP - Vision-language model with diffusion policy
# eval_policy "configs/policy/qwen2dp.yaml"

# Qwen2.5DP - Updated Qwen2.5-VL model
# eval_policy "configs/policy/qwen25dp.yaml"

# Diffusion Policy - Pure diffusion-based approach
# eval_policy "configs/policy/diffusion_policy.yaml"

# DiVLA - Vision-language diffusion policy
# eval_policy "configs/policy/divla.yaml"

# RDT - Robotic Decision Transformer
# eval_policy "configs/policy/rdt.yaml"

echo "Evaluation completed!"
