# OpenVLA Policy Integration

This document describes the OpenVLA policy integration in the IL-Studio repository.

## Overview

OpenVLA (Open Vision-Language-Action) is a vision-language model that can be used for robot action prediction. This integration provides support for both LoRA fine-tuning and full fine-tuning modes.

## Features

- **Training Modes**: Support for both LoRA (Low-Rank Adaptation) and full fine-tuning
- **Quantization**: Optional 4-bit quantization for LoRA training (memory efficient)
- **Flexible Configuration**: YAML-based configuration system
- **Integration**: Seamless integration with existing training and evaluation pipelines

## Configuration

The OpenVLA policy is configured via `configs/policy/openvla.yaml`:

```yaml
name: openvla
module_path: policy.openvla
config_class: OpenVLAPolicyConfig
model_class: OpenVLAPolicy
pretrained_config:
  model_name_or_path: "microsoft/git-base"  # Replace with actual OpenVLA model
  is_pretrained: true
config_params:
  # Training mode: "lora" or "full"
  training_mode: "lora"
  # LoRA parameters (only used when training_mode="lora")
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # Quantization (only used when training_mode="lora")
  use_quantization: false
  # Model parameters
  max_length: 2048
  # Task parameters
  state_dim: 14
  action_dim: 14
  camera_names: ["primary"]
```

### Configuration Parameters

- **training_mode**: Choose between "lora" (recommended) or "full" fine-tuning
- **lora_r**: LoRA rank (higher = more parameters, default: 16)
- **lora_alpha**: LoRA alpha parameter (default: 32)
- **lora_dropout**: LoRA dropout rate (default: 0.1)
- **use_quantization**: Enable 4-bit quantization for LoRA (saves memory)
- **max_length**: Maximum sequence length for text processing
- **state_dim**: Robot state dimension
- **action_dim**: Robot action dimension
- **camera_names**: List of camera names for input

## Usage

### Training

Train an OpenVLA model using the provided script:

```bash
# Make the script executable
chmod +x scripts/train_openvla_example.sh

# Run training
./scripts/train_openvla_example.sh
```

Or use the training script directly:

```bash
python train.py \
    --policy_config configs/policy/openvla.yaml \
    --task_config configs/task/sim_transfer_cube_scripted.yaml \
    --data_dir data/sim_transfer_cube_scripted \
    --output_dir ckpt/openvla_example \
    --max_steps 2000 \
    --training_mode lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1
```

### Evaluation

Evaluate a trained OpenVLA model:

```bash
# Make the script executable
chmod +x scripts/eval_openvla_example.sh

# Run evaluation
./scripts/eval_openvla_example.sh
```

Or use the evaluation script directly:

```bash
python eval.py \
    --model_name_or_path ckpt/openvla_example \
    --task_config configs/task/sim_transfer_cube_scripted.yaml \
    --data_dir data/sim_transfer_cube_scripted \
    --output_dir results/openvla_eval
```

### Real Robot Evaluation

For real robot evaluation:

```bash
python eval_real.py \
    --model_name_or_path ckpt/openvla_example \
    --task_config configs/task/sim_transfer_cube_scripted.yaml \
    --robot_config configs/robot/agilex_aloha.yaml
```

## Model Requirements

### Current Setup

The current configuration uses `microsoft/git-base` as a placeholder model. To use the actual OpenVLA model:

1. **Download the OpenVLA model** from the official repository
2. **Update the model path** in `configs/policy/openvla.yaml`:
   ```yaml
   pretrained_config:
     model_name_or_path: "/path/to/openvla-model"  # Local path
     is_pretrained: true
   ```

### Alternative Models

You can use other vision-language models that support:
- Image and text inputs
- Text generation
- Hugging Face transformers compatibility

Examples:
- `microsoft/git-base`
- `Salesforce/blip2-opt-2.7b`
- `Salesforce/instructblip-vicuna-7b`

## Training Modes

### LoRA Training (Recommended)

LoRA (Low-Rank Adaptation) is the recommended training mode because:
- **Memory Efficient**: Uses much less GPU memory
- **Faster Training**: Trains only a small subset of parameters
- **Good Performance**: Often achieves similar results to full fine-tuning

```bash
python train.py \
    --policy_config configs/policy/openvla.yaml \
    --training_mode lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1
```

### Full Fine-tuning

Full fine-tuning trains all model parameters:

```bash
python train.py \
    --policy_config configs/policy/openvla.yaml \
    --training_mode full
```

### Quantized LoRA Training

For even more memory efficiency, use 4-bit quantization:

```bash
python train.py \
    --policy_config configs/policy/openvla.yaml \
    --training_mode lora \
    --use_quantization true
```

## Data Format

The OpenVLA policy expects data in the following format:

```python
{
    'image': torch.Tensor,      # Shape: (C, H, W) or (B, C, H, W)
    'action': torch.Tensor,     # Shape: (action_dim,) or (B, action_dim)
    'raw_lang': str            # Natural language instruction
}
```

## Implementation Details

### Model Architecture

The OpenVLA policy uses:
- **Vision Encoder**: Processes input images
- **Language Model**: Handles text processing and generation
- **Action Head**: Converts language model outputs to robot actions

### Data Processing

1. **Image Processing**: Images are processed through the vision encoder
2. **Text Processing**: Instructions are tokenized and processed
3. **Action Generation**: The model generates text descriptions of actions
4. **Action Parsing**: Text descriptions are converted to numerical actions

### Training Process

1. **Data Loading**: Load and preprocess training data
2. **Model Loading**: Load pretrained vision-language model
3. **LoRA Setup**: Apply LoRA adapters (if using LoRA mode)
4. **Training Loop**: Standard transformer training with language modeling loss
5. **Evaluation**: Periodic evaluation on validation set
6. **Saving**: Save model checkpoints and final model

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure the model path is correct and the model is accessible
2. **Memory Issues**: Use LoRA training or quantization to reduce memory usage
3. **Import Errors**: Ensure all required dependencies are installed
4. **CUDA Issues**: Check GPU availability and CUDA installation

### Memory Requirements

- **Full Fine-tuning**: ~24GB GPU memory (for 7B model)
- **LoRA Training**: ~8-12GB GPU memory
- **Quantized LoRA**: ~4-6GB GPU memory

### Performance Tips

1. **Use LoRA**: Recommended for most use cases
2. **Batch Size**: Start with small batch sizes and increase gradually
3. **Learning Rate**: Use lower learning rates for LoRA (1e-4 to 1e-5)
4. **Gradient Accumulation**: Use gradient accumulation for larger effective batch sizes

## Examples

See the example scripts in the `scripts/` directory:
- `train_openvla_example.sh`: Training example
- `eval_openvla_example.sh`: Evaluation example

## Future Improvements

- [ ] Support for more vision-language models
- [ ] Better action parsing and generation
- [ ] Multi-modal data augmentation
- [ ] Advanced training strategies (FSDP, DeepSpeed)
- [ ] Real-time inference optimization
