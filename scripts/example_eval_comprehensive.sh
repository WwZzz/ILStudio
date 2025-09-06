#!/bin/bash
# Comprehensive evaluation example using the new policy configuration system

export MUJOCO_GL=glfw
export DM_CONTROL_RENDERER=software
export NUMPY_EXPERIMENTAL_DTYPE_API=1

# Configuration variables
ENV=aloha
TASKNAME=transfer_cube_top
CHUNKSIZE=100
DATASET=data/sim_transfer_cube_scripted
FPS=50
ROLLOUT=10
PARALLEL=5

# Function to evaluate a policy
eval_policy() {
    local policy_config=$1
    local policy_name=$(basename "$policy_config" .yaml)
    local ckpt_path=$2
    local output_dir=results/${policy_name}_${TASKNAME}_${CHUNKSIZE}_example
    
    echo "Evaluating policy: $policy_config"
    echo "Checkpoint: $ckpt_path"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    python eval.py --env_name $ENV \
        --task $TASKNAME \
        --policy_config $policy_config \
        --model_name_or_path $ckpt_path \
        --save_dir $output_dir \
        --num_rollout $ROLLOUT \
        --num_envs $PARALLEL \
        --dataset_dir $DATASET \
        --fps $FPS \
        --freq $CHUNKSIZE \
        --chunk_size $CHUNKSIZE \
        --camera_ids "[0]" \
    --image_size_primary "(640,480)" \
    --image_size_wrist "(256,256)" \
    --max_timesteps 400
}

echo "Policy Evaluation Examples"
echo "=========================="
echo "Environment: $ENV"
echo "Task: $TASKNAME"
echo "Chunk Size: $CHUNKSIZE"
echo "Rollouts: $ROLLOUT"
echo "Parallel Environments: $PARALLEL"
echo ""

# Example 1: ACT Policy
echo "1. Evaluating ACT Policy"
eval_policy "configs/policy/act.yaml" "/home/noematrix/Desktop/IL-Studio/ckpt/act_sim_transfer_cube_scripted_zscore_example/checkpoint-100"

# Example 2: Diffusion Policy (commented out - uncomment to use)
# echo "2. Evaluating Diffusion Policy"
# eval_policy "configs/policy/diffusion_policy.yaml" "/path/to/diffusion_policy_checkpoint"

# Example 3: Qwen2DP Policy (commented out - uncomment to use)
# echo "3. Evaluating Qwen2DP Policy"
# eval_policy "configs/policy/qwen2dp.yaml" "/path/to/qwen2dp_checkpoint"

echo ""
echo "Evaluation completed!"
echo ""
echo "To evaluate other policies:"
echo "1. Uncomment the desired policy in this script"
echo "2. Update the checkpoint path to point to your trained model"
echo "3. Run: bash scripts/example_eval_comprehensive.sh"
