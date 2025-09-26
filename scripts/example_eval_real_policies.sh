#!/bin/bash

# Example real-world evaluation script using the new policy config system
# This script demonstrates how to evaluate different policies on real robots

export MUJOCO_GL=glfw
export DM_CONTROL_RENDERER=software
export NUMPY_EXPERIMENTAL_DTYPE_API=1

ROBOT_CONFIG=configs/robots/dummy.yaml
TASKNAME=sim_transfer_cube_scripted
CHUNK_SIZE=100
PUBLISH_RATE=25
SENSING_RATE=20
SAVE_DIR=results/real_eval_policies

echo "Real-World Policy Evaluation Examples"
echo "====================================="

# Function to evaluate a policy on real robot
eval_real_policy() {
    local policy_config_path=$1
    local policy_name=$(basename "$policy_config_path" .yaml)
    local output_dir=${SAVE_DIR}/${policy_name}_${TASKNAME}
    
    echo "Evaluating policy on real robot: $policy_config_path"
    echo "Output directory: $output_dir"
    echo "Robot config: $ROBOT_CONFIG"
    
    python eval_real.py \
        --policy_config $policy_config_path \
        --robot_config $ROBOT_CONFIG \
        --task $TASKNAME \
        --chunk_size $CHUNK_SIZE \
        --publish_rate $PUBLISH_RATE \
        --sensing_rate $SENSING_RATE \
        --save_dir $output_dir \
        --max_timesteps 400 \
        --fps 50 \
        --image_size "(640,480)" \
        --ctrl_space joint \
        --ctrl_type abs \
        --camera_ids "[0]" \
        --camera_names primary \
        --action_manager OlderFirstManager \
        --manager_coef 1.0
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

echo "Available robot configs:"
echo "  - configs/robots/dummy.yaml (simulation)"
echo "  - configs/robots/agilex_aloha.yaml (real robot)"
echo "  - configs/robots/franka_panda_sim_pybullet.yaml (PyBullet simulation)"
echo ""

# Uncomment the policy you want to evaluate:

# ACT - Fast and efficient transformer-based policy
eval_real_policy "configs/policy/act.yaml"

# Qwen2DP - Vision-language model with diffusion policy
# eval_real_policy "configs/policy/qwen2dp.yaml"

# Qwen2.5DP - Updated Qwen2.5-VL model
# eval_real_policy "configs/policy/qwen25dp.yaml"

# Diffusion Policy - Pure diffusion-based approach
# eval_real_policy "configs/policy/diffusion_policy.yaml"

# DiVLA - Vision-language diffusion policy
# eval_real_policy "configs/policy/divla.yaml"

# RDT - Robotic Decision Transformer
# eval_real_policy "configs/policy/rdt.yaml"

echo "Real-world evaluation completed!"
