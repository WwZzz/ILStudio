#!/bin/bash

# Example script for evaluating OpenVLA policy
# This script demonstrates how to use the new simplified policy loader design for evaluation

set -e  # Exit on any error

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide the checkpoint path as an argument"
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ckpt/openvla_sim_transfer_cube_scripted/checkpoint-1000"
    exit 1
fi

CHECKPOINT_PATH=$1

# Validate checkpoint path
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint directory does not exist: $CHECKPOINT_PATH"
    exit 1
fi

echo "🔍 Starting OpenVLA evaluation..."
echo "📁 Checkpoint: $CHECKPOINT_PATH"

# Activate the environment
source /home/wz/miniconda3/etc/profile.d/conda.sh
conda activate ilstd

# Evaluation configuration
TASK_NAME="sim_transfer_cube_scripted"
NUM_ROLLOUTS=5
MAX_TIMESTEPS=200

# Evaluation arguments
EVAL_ARGS="
--model_name_or_path $CHECKPOINT_PATH
--task $TASK_NAME
--num_rollout $NUM_ROLLOUTS
--max_timesteps $MAX_TIMESTEPS
--is_pretrained
--device cuda
--save_dir results/openvla_eval_$(basename $CHECKPOINT_PATH)
--fps 10
--num_envs 1
--ctrl_space joint
--ctrl_type pos
--camera_ids 0
"

echo "📋 Evaluation Configuration:"
echo "  Task: $TASK_NAME"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Number of Rollouts: $NUM_ROLLOUTS"
echo "  Max Timesteps: $MAX_TIMESTEPS"
echo "  Device: CUDA"
echo "  Control Space: Joint Position"
echo ""

# Create results directory
mkdir -p results/openvla_eval_$(basename $CHECKPOINT_PATH)

# Run evaluation
echo "🎯 Starting evaluation..."
python eval.py $EVAL_ARGS

echo "✅ Evaluation completed successfully!"
echo "📊 Results saved to: results/openvla_eval_$(basename $CHECKPOINT_PATH)"
echo ""
echo "📈 To view results, check the saved files in the results directory"
