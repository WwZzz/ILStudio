#!/bin/bash

# Example script for training OpenVLA policy on sim_transfer_cube_scripted task
# This script demonstrates how to use the new simplified policy loader design

set -e  # Exit on any error

echo "🚀 Starting OpenVLA training on sim_transfer_cube_scripted task..."

# Activate the environment
source /home/wz/miniconda3/etc/profile.d/conda.sh
conda activate ilstd

# Training configuration
TASK_NAME="sim_transfer_cube_scripted"
POLICY_CONFIG="configs/policy/openvla.yaml"
TRAINING_CONFIG="configs/training/openvla_finetune.yaml"
OUTPUT_DIR="ckpt/openvla_sim_transfer_cube_scripted"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training arguments (only the basic arguments that train.py accepts)
TRAINING_ARGS="
--task_name $TASK_NAME
--policy_config $POLICY_CONFIG
--training_config $TRAINING_CONFIG
--output_dir $OUTPUT_DIR
"

echo "📋 Training Configuration:"
echo "  Task: $TASK_NAME"
echo "  Policy Config: $POLICY_CONFIG"
echo "  Training Config: $TRAINING_CONFIG"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model: openvla/openvla-7b (configured in policy config)"
echo "  Training Mode: LoRA (configured in policy config)"
echo "  Max Steps: 1000 (configured in training config)"
echo "  Batch Size: 2 (configured in training config)"
echo "  Learning Rate: 1e-4 (configured in training config)"
echo ""

# Run training
echo "🏋️ Starting training..."
python train.py $TRAINING_ARGS

echo "✅ Training completed successfully!"
echo "📁 Checkpoint saved to: $OUTPUT_DIR"
echo ""
echo "🔍 To evaluate the trained model, run:"
echo "   bash scripts/example_eval_openvla.sh $OUTPUT_DIR/checkpoint-1000"
