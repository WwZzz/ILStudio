#!/bin/bash

# Example training script for OpenVLA policy
# This script demonstrates how to train OpenVLA with different configurations

set -e

# Configuration
POLICY_CONFIG="configs/policy/openvla.yaml"
TASK_CONFIG="configs/task/sim_transfer_cube_scripted.yaml"
OUTPUT_DIR="ckpt/openvla_sim_transfer_cube_scripted_example"
DATA_DIR="data/sim_transfer_cube_scripted"

# Training parameters
MAX_STEPS=2000
BATCH_SIZE=4
LEARNING_RATE=1e-4
SAVE_STEPS=500
EVAL_STEPS=500
LOGGING_STEPS=100

# LoRA parameters (for LoRA training mode)
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Quantization (optional, only for LoRA mode)
USE_QUANTIZATION=false

echo "🚀 Starting OpenVLA training..."
echo "Policy config: $POLICY_CONFIG"
echo "Task config: $TASK_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Please ensure the data is available before training."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Training command
python train.py \
    --policy_config "$POLICY_CONFIG" \
    --task_config "$TASK_CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --report_to "tensorboard" \
    --run_name "openvla_sim_transfer_cube_scripted" \
    --overwrite_output_dir \
    --training_mode "lora" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --use_quantization "$USE_QUANTIZATION"

echo ""
echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "📊 To view training logs:"
echo "   tensorboard --logdir logs"
echo ""
echo "🔍 To evaluate the model:"
echo "   python eval.py --model_name_or_path $OUTPUT_DIR --task_config $TASK_CONFIG"
