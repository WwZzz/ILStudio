# OpenVLA Fine-tuning for sim_transfer_cube_scripted

This guide provides step-by-step instructions for fine-tuning OpenVLA 7B model on the `sim_transfer_cube_scripted` task.

## Overview

OpenVLA (Open Vision-Language-Action) is a 7B parameter vision-language model designed for robot action prediction. This guide shows how to fine-tune it on the simulated transfer cube task using LoRA (Low-Rank Adaptation) for efficient training.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended: A100, V100, or RTX 4090)
- **RAM**: At least 32GB system RAM
- **Storage**: At least 50GB free space for model weights and data

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- All dependencies from `requirements.txt`

## Quick Start

### 1. Prepare Data

First, ensure you have the `sim_transfer_cube_scripted` dataset:

```bash
# If you don't have the data, collect it first
python collect_data.py --task sim_transfer_cube_scripted --num_episodes 100

# Prepare data for OpenVLA format
python prepare_openvla_data.py \
    --input_dir data/sim_transfer_cube_scripted \
    --output_dir data/openvla_sim_transfer_cube_scripted
```

### 2. Start Fine-tuning

Run the fine-tuning script:

```bash
./scripts/train_openvla_sim_transfer_cube.sh
```

This will:
- Download the OpenVLA 7B model from Hugging Face
- Fine-tune using LoRA with optimized hyperparameters
- Save checkpoints every 500 steps
- Log training progress to TensorBoard

### 3. Monitor Training

Monitor training progress:

```bash
# View TensorBoard logs
tensorboard --logdir logs/

# Check training logs
tail -f logs/train.log
```

### 4. Evaluate Model

After training, evaluate the model:

```bash
./scripts/eval_openvla_sim_transfer_cube.sh
```

## Detailed Configuration

### Model Configuration

The OpenVLA model is configured in `configs/policy/openvla.yaml`:

```yaml
pretrained_config:
  model_name_or_path: "openvla/openvla-7b"  # Official OpenVLA 7B model
  is_pretrained: true

config_params:
  training_mode: "lora"           # Use LoRA for efficient fine-tuning
  lora_r: 32                      # LoRA rank (higher = more parameters)
  lora_alpha: 64                  # LoRA alpha parameter
  lora_dropout: 0.1               # LoRA dropout rate
  use_quantization: false         # Disable quantization for better performance
  max_length: 2048                # Maximum sequence length
  state_dim: 14                   # Robot state dimension
  action_dim: 14                  # Robot action dimension
  camera_names: ["primary"]       # Camera names for input
```

### Training Configuration

Training parameters are in `configs/training/openvla_finetune.yaml`:

```yaml
# Core training parameters
max_steps: 5000                   # Total training steps
learning_rate: 5e-4               # Learning rate
per_device_train_batch_size: 8    # Batch size per GPU
gradient_accumulation_steps: 2    # Gradient accumulation
warmup_steps: 100                 # Warmup steps
lr_scheduler_type: "cosine"       # Learning rate scheduler

# LoRA parameters
lora_r: 32                        # LoRA rank
lora_alpha: 64                    # LoRA alpha
lora_dropout: 0.1                 # LoRA dropout

# Memory optimization
fp16: true                        # Mixed precision training
gradient_checkpointing: true      # Gradient checkpointing
```

## Advanced Usage

### Custom Hyperparameters

You can override training parameters:

```bash
python train.py \
    --task_name sim_transfer_cube_scripted \
    --policy_config configs/policy/openvla.yaml \
    --training_config configs/training/openvla_finetune.yaml \
    --output_dir ckpt/my_openvla_model \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --lora_r 64 \
    --lora_alpha 128
```

### Full Fine-tuning

For full fine-tuning (requires more memory):

```bash
python train.py \
    --task_name sim_transfer_cube_scripted \
    --policy_config configs/policy/openvla.yaml \
    --training_config configs/training/openvla_finetune.yaml \
    --output_dir ckpt/openvla_full_finetuned \
    --training_mode full \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### Quantized Training

For memory-constrained environments:

```bash
python train.py \
    --task_name sim_transfer_cube_scripted \
    --policy_config configs/policy/openvla.yaml \
    --training_config configs/training/openvla_finetune.yaml \
    --output_dir ckpt/openvla_quantized \
    --use_quantization true \
    --per_device_train_batch_size 16
```

## Data Format

The OpenVLA fine-tuning expects data in the following format:

```
data/openvla_sim_transfer_cube_scripted/
├── episodes/
│   ├── episode_000000.npz
│   ├── episode_000001.npz
│   └── ...
├── metadata.json
└── dataset_config.json
```

Each episode file contains:
- `observations`: Dictionary with images, qpos, qvel, etc.
- `actions`: Array of actions (shape: [T, 14])
- `rewards`: Array of rewards
- `dones`: Array of done flags

## Monitoring and Debugging

### Training Metrics

Key metrics to monitor:
- **Loss**: Should decrease steadily
- **Learning Rate**: Follows cosine schedule
- **Gradient Norm**: Should be stable
- **Memory Usage**: Should be within GPU limits

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable `use_quantization: true`

2. **Slow Training**:
   - Increase `dataloader_num_workers`
   - Enable `fp16: true`
   - Use faster storage (SSD)

3. **Poor Performance**:
   - Increase `lora_r` for more parameters
   - Adjust learning rate
   - Check data quality

### Debugging

Enable debug mode:

```bash
python train.py \
    --task_name sim_transfer_cube_scripted \
    --policy_config configs/policy/openvla.yaml \
    --training_config configs/training/openvla_finetune.yaml \
    --output_dir ckpt/debug \
    --debug_mode true \
    --logging_steps 1
```

## Evaluation

### Simulation Evaluation

```bash
./scripts/eval_openvla_sim_transfer_cube.sh
```

### Real Robot Evaluation

```bash
python eval_real.py \
    --model_name_or_path ckpt/openvla_sim_transfer_cube_scripted_finetuned \
    --task_config configs/task/sim_transfer_cube_scripted.yaml \
    --robot_config configs/robot/agilex_aloha.yaml
```

### Metrics

The evaluation provides:
- **Success Rate**: Percentage of successful episodes
- **Episode Length**: Average episode duration
- **Action Accuracy**: Mean squared error of actions
- **Task Completion**: Specific task metrics

## Model Deployment

### Export Model

Export the fine-tuned model:

```bash
python export_model.py \
    --model_path ckpt/openvla_sim_transfer_cube_scripted_finetuned \
    --output_path models/openvla_sim_transfer_cube_scripted_exported
```

### Inference

Use the model for inference:

```python
from policy.openvla import OpenVLAPolicy, OpenVLAPolicyConfig

# Load model
config = OpenVLAPolicyConfig.from_pretrained("ckpt/openvla_sim_transfer_cube_scripted_finetuned")
model = OpenVLAPolicy.from_pretrained("ckpt/openvla_sim_transfer_cube_scripted_finetuned")

# Predict action
action = model.predict_action(image, "Transfer the red cube to the other arm.")
```

## Troubleshooting

### Common Problems

1. **Model Loading Issues**:
   - Check internet connection for Hugging Face download
   - Verify model path in config
   - Check CUDA compatibility

2. **Data Loading Issues**:
   - Verify data format with `prepare_openvla_data.py`
   - Check file permissions
   - Validate data integrity

3. **Training Issues**:
   - Monitor GPU memory usage
   - Check learning rate schedule
   - Verify data preprocessing

### Getting Help

- Check the logs in `logs/` directory
- Use TensorBoard for visualization
- Review the test scripts for examples
- Check the OpenVLA documentation

## Performance Tips

### Training Speed
- Use multiple GPUs with `torchrun`
- Enable mixed precision training (`fp16: true`)
- Use fast storage (NVMe SSD)
- Increase `dataloader_num_workers`

### Memory Efficiency
- Use LoRA instead of full fine-tuning
- Enable gradient checkpointing
- Use quantization for very large models
- Reduce batch size if needed

### Model Quality
- Use higher LoRA rank for more parameters
- Adjust learning rate schedule
- Ensure high-quality training data
- Use appropriate data augmentation

## Next Steps

After successful fine-tuning:

1. **Evaluate Performance**: Run comprehensive evaluation
2. **Deploy Model**: Use for real robot control
3. **Iterate**: Collect more data and retrain if needed
4. **Scale**: Apply to other manipulation tasks

## References

- [OpenVLA Paper](https://arxiv.org/abs/2310.03127)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
