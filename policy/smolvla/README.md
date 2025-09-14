# SmolVLA Policy Integration

This directory contains the SmolVLA policy integration for the IL-Studio framework, following the established pattern of `load_model`, `get_data_processor`, `get_data_collator`, and `trainer` separation.

## Files Structure

- `configuration.py` - SmolVLA configuration class based on lerobot's SmolVLAConfig
- `modeling.py` - Main SmolVLA model implementation without PreTrainedPolicy dependency
- `smolvlm_with_expert.py` - SmolVLM with action expert model
- `data_utils.py` - Data processing utilities and collators
- `trainer.py` - Custom trainer for SmolVLA training
- `__init__.py` - Module exports and main interface functions

## Key Features

### 1. Configuration (`SmolVLAConfig`)
- Based on lerobot's SmolVLAConfig but adapted for this framework
- Supports all major SmolVLA parameters including VLM model selection, training settings, and task parameters
- Includes proper feature validation and normalization mapping

### 2. Model (`SmolVLAPolicy`)
- Implements the main SmolVLA policy without depending on lerobot's PreTrainedPolicy
- Includes VLAFlowMatching model for flow-based action generation
- Supports both training and inference modes
- Handles image preprocessing, language tokenization, and action prediction

### 3. Data Processing
- `SmolVLAProcessor` for processing individual samples
- `data_collator` for batching samples
- Support for multi-camera inputs and text instructions
- Proper image normalization and tokenization

### 4. Training
- `SmolVLATrainer` extends transformers Trainer for SmolVLA-specific training
- Custom loss computation and prediction steps
- Support for evaluation metrics

## Usage

### Loading the Policy
```python
from policy.policy_loader import PolicyLoader

loader = PolicyLoader()
model_components = loader.load_model("configs/policy/smolvla.yaml", args)
```

### Training
```python
from policy.smolvla import create_smolvla_trainer

trainer = create_smolvla_trainer(
    model_name_or_path="path/to/checkpoint",
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)
```

### Configuration
The policy can be configured via the YAML file at `configs/policy/smolvla.yaml`:

```yaml
name: "smolvla"
module_path: "policy.smolvla"
config_class: "SmolVLAConfig"
model_class: "SmolVLAPolicy"
data_processor: "get_data_processor"
data_collator: "get_data_collator"
trainer_class: "SmolVLATrainer"
```

## Dependencies

The SmolVLA policy requires the following dependencies:
- `torch`
- `transformers`
- `safetensors`
- `PIL` (Pillow)
- `torchvision`

## Integration with Framework

This implementation follows the established pattern in the IL-Studio framework:

1. **Separation of Concerns**: Model, data processing, and training are separated into different modules
2. **Configuration-driven**: Uses YAML configuration files for easy parameter adjustment
3. **Policy Loader Compatible**: Works with the existing policy loading system
4. **No PreTrainedPolicy Dependency**: Avoids lerobot's PreTrainedPolicy to maintain framework independence

## Key Differences from LeRobot Implementation

1. **No PreTrainedPolicy**: Removed dependency on lerobot's PreTrainedPolicy base class
2. **Simplified Architecture**: Focused on core functionality without lerobot-specific features
3. **Framework Integration**: Designed to work with IL-Studio's policy loading and training systems
4. **Configuration**: Uses standard transformers PretrainedConfig instead of lerobot's config system

## Testing

The integration can be tested by loading the policy configuration:

```python
from policy.policy_loader import PolicyLoader

loader = PolicyLoader()
config = loader.load_policy_config("configs/policy/smolvla.yaml")
print(f"Policy: {config.name}, Module: {config.module_path}")
```

This confirms that the policy is properly integrated with the framework's configuration system.
