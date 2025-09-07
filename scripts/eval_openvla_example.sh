#!/bin/bash

# Example evaluation script for OpenVLA policy
# This script demonstrates how to evaluate OpenVLA models

set -e

# Configuration
MODEL_PATH="ckpt/openvla_sim_transfer_cube_scripted_example"  # Path to trained model
TASK_CONFIG="configs/task/sim_transfer_cube_scripted.yaml"
OUTPUT_DIR="results/openvla_sim_transfer_cube_scripted_eval"
DATA_DIR="data/sim_transfer_cube_scripted"

# Evaluation parameters
BATCH_SIZE=4
NUM_EPISODES=10
RENDER_VIDEO=true
SAVE_VIDEOS=true

echo "🔍 Starting OpenVLA evaluation..."
echo "Model path: $MODEL_PATH"
echo "Task config: $TASK_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    echo "Please train a model first or provide the correct path."
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Please ensure the data is available before evaluation."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Evaluation command
python eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --task_config "$TASK_CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --num_episodes "$NUM_EPISODES" \
    --render_video "$RENDER_VIDEO" \
    --save_videos "$SAVE_VIDEOS" \
    --dataloader_num_workers 4 \
    --remove_unused_columns false

echo ""
echo "✅ Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 To view evaluation results:"
echo "   ls -la $OUTPUT_DIR"
echo ""
echo "🎥 To view videos (if generated):"
echo "   ls -la $OUTPUT_DIR/videos"
