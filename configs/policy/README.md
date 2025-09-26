# Policy Configurations

This directory contains YAML configuration files for different policy models. Each YAML file defines the model architecture, configuration parameters, and data processing components.

## Available Policies

### ACT (Action Chunking with Transformers)
- **File**: `act.yaml`
- **Module**: `policy.act`
- **Description**: Transformer-based policy for action chunking
- **Key Features**: 
  - Configurable chunk size and prediction horizon
  - Support for multiple camera inputs
  - History length configuration

### Qwen2DP (Qwen2 Vision-Language Diffusion Policy)
- **File**: `qwen2dp.yaml`
- **Module**: `policy.qwen2dp`
- **Description**: Vision-language model with diffusion policy
- **Key Features**:
  - Qwen2-VL backbone
  - LoRA fine-tuning support
  - Multimodal processing

### Qwen2.5DP (Qwen2.5 Vision-Language Diffusion Policy)
- **File**: `qwen25dp.yaml`
- **Module**: `policy.qwen25dp`
- **Description**: Updated Qwen2.5-VL model with diffusion policy
- **Key Features**:
  - Qwen2.5-VL backbone
  - Enhanced vision-language capabilities
  - LoRA fine-tuning support

### Diffusion Policy
- **File**: `diffusion_policy.yaml`
- **Module**: `policy.diffusion_policy`
- **Description**: Pure diffusion-based policy
- **Key Features**:
  - DDPM noise scheduler
  - Configurable observation and action encoders
  - Temporal action modeling

### DiVLA (Diffusion Vision-Language Action)
- **File**: `divla.yaml`
- **Module**: `policy.divla`
- **Description**: Vision-language diffusion policy
- **Key Features**:
  - CLIP vision encoder
  - Qwen2-VL language model
  - Cross-attention fusion

### RDT (Robotic Decision Transformer)
- **File**: `rdt.yaml`
- **Module**: `policy.rdt`
- **Description**: Transformer-based decision making
- **Key Features**:
  - ResNet18 observation encoder
  - Temporal action modeling
  - Pretrained weight support

## Configuration Format

Each YAML file follows this structure:

```yaml
name: policy_name                    # Policy identifier
module_path: policy.module_name        # Python module path
config_class: ConfigClassName       # Configuration class name
model_class: ModelClassName         # Model class name
pretrained_config:                  # Pretrained model configuration
  model_name_or_path: "path/to/model"
  is_pretrained: false
config_params:                      # Config class initialization parameters
  backbone: "resnet18"
  hidden_dim: 512
  enc_layers: 4
  dec_layers: 7
  dropout: 0.1
  lr_backbone: 0.00001
  # ... other config parameters
model_args:                         # Model-specific arguments
  action_dim: 14
  state_dim: 14
  # ... other model parameters
data_processor: function_name       # Data processing function (optional)
data_collator: function_name        # Data collator function (optional)
trainer_class: TrainerClassName     # Trainer class (optional)
```

### Config Parameters

The `config_params` section allows you to specify all parameters needed to initialize the config class. These parameters will be used to create a config instance, which can then be overridden by command-line arguments if provided.

**Key Features:**
- **YAML Parameters**: All config class parameters can be specified in YAML
- **Args Override**: Command-line arguments override YAML parameters
- **Type Safety**: Parameters are passed directly to the config class constructor
- **Flexibility**: Easy to modify model architecture without changing code

## Usage

### In Training Scripts

```python
from policy.policy_loader import load_policy_model_with_config, get_policy_data_processor, get_policy_data_collator

# Load model using policy configuration with YAML parameters
model_components = load_policy_model_with_config("act", args)
model = model_components['model']
config = model_components['config']  # Config instance created from YAML

# Get data processor
data_processor = get_policy_data_processor("act", dataset, args, model_components)

# Get data collator
data_collator = get_policy_data_collator("act", args, model_components)
```

### Command Line Usage

```bash
# Train with ACT policy (using full path)
python train.py --policy_config configs/policy/act.yaml --task_name sim_transfer_cube_scripted

# Train with Qwen2DP policy (using full path)
python train.py --policy_config configs/policy/qwen2dp.yaml --task_name sim_transfer_cube_scripted

# Evaluation
python eval.py --policy_config configs/policy/act.yaml --env_name aloha --task sim_transfer_cube_scripted

# Real-world evaluation
python eval_real.py --policy_config configs/policy/act.yaml --robot_config configs/robots/dummy.yaml --task sim_transfer_cube_scripted
```

## Adding New Policies

To add a new policy:

1. Create a YAML file in this directory
2. Define the module path and configuration parameters
3. Ensure the module implements the required `load_model` function
4. Optionally implement `get_data_processor` and `get_data_collator` functions
5. Test the configuration with the training script

## Benefits

- **Modularity**: Each policy is self-contained in its configuration
- **Flexibility**: Easy to add new policies without code changes
- **Maintainability**: Configuration parameters are clearly separated
- **Reusability**: Same policy can be used across different tasks
- **Version Control**: Policy configurations can be tracked separately
