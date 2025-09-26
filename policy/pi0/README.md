# PI0 Policy Integration

This directory contains the PI0 policy integration for the IL-Studio framework, following the established pattern of `load_model`, `get_data_processor`, `get_data_collator`, and `trainer` separation.

## Files Structure

- `modeling_pi0.py` - Main PI0 model implementation with flow matching
- `paligemma_with_expert.py` - PaliGemma with action expert model
- `data_utils.py` - Data processing utilities and collators
- `trainer.py` - Custom trainer for PI0 training
- `__init__.py` - Module exports and main interface functions

## Key Features

### 1. Configuration (`PI0FlowMatchingConfig`)
- Based on the original PI0FlowMatchingConfig but enhanced for this framework
- Supports all major PI0 parameters including projection width, training settings, and task parameters
- Includes proper feature validation and normalization mapping
- Supports both standard and fast variants

### 2. Model (`PI0FlowMatching`)
- Implements the main PI0 policy with flow matching for action generation
- Uses PaliGemma as the vision-language backbone
- Supports both training and inference modes
- Handles image preprocessing, language tokenization, and action prediction

### 3. Data Processing
- `PI0Processor` for processing individual samples
- `data_collator` for batching samples
- Support for multi-camera inputs and text instructions
- Proper image normalization and tokenization

### 4. Training
- `PI0Trainer` extends transformers Trainer for PI0-specific training
- Custom loss computation and prediction steps
- Support for evaluation metrics

## Usage

### Loading the Policy
```python
from policy.policy_loader import PolicyLoader

loader = PolicyLoader()
model_components = loader.load_model("configs/policy/pi0.yaml", args)
```

### Training
```python
from policy.pi0 import create_pi0_trainer

trainer = create_pi0_trainer(
    model_name_or_path="path/to/checkpoint",
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)
```

### Configuration
The policy can be configured via YAML files:

- `configs/policy/pi0.yaml` - Standard PI0 configuration
- `configs/policy/pi0fast.yaml` - Optimized for speed with smaller chunk size and fewer denoising steps

```yaml
name: "pi0"
module_path: "policy.pi0"
config_class: "PI0FlowMatchingConfig"
model_class: "PI0FlowMatching"
data_processor: "get_data_processor"
data_collator: "get_data_collator"
trainer_class: "PI0Trainer"
```

## Variants

### PI0 (Standard)
- Chunk size: 50
- Projection width: 1024
- Denoising steps: 10
- Tokenizer max length: 48

### PI0Fast
- Chunk size: 25 (faster inference)
- Projection width: 512 (smaller model)
- Denoising steps: 5 (faster generation)
- Tokenizer max length: 32 (faster processing)
- Eager attention implementation

## Dependencies

The PI0 policy requires the following dependencies:
- `torch`
- `transformers`
- `PIL` (Pillow)
- `torchvision`

## Integration with Framework

This implementation follows the established pattern in the IL-Studio framework:

1. **Separation of Concerns**: Model, data processing, and training are separated into different modules
2. **Configuration-driven**: Uses YAML configuration files for easy parameter adjustment
3. **Policy Loader Compatible**: Works with the existing policy loading system
4. **Dual Variants**: Supports both standard and fast versions for different use cases

## Key Differences from Original Implementation

1. **Enhanced Configuration**: Added feature validation and normalization mapping
2. **Framework Integration**: Designed to work with IL-Studio's policy loading and training systems
3. **Dual Variants**: Created both standard and fast versions
4. **Improved Data Processing**: Better integration with the framework's data handling

## Testing

The integration can be tested by loading the policy configuration:

```python
from policy.policy_loader import PolicyLoader

loader = PolicyLoader()
config = loader.load_policy_config("configs/policy/pi0.yaml")
print(f"Policy: {config.name}, Module: {config.module_path}")

# Test PI0Fast variant
config_fast = loader.load_policy_config("configs/policy/pi0fast.yaml")
print(f"Policy: {config_fast.name}, Module: {config_fast.module_path}")
```

This confirms that both PI0 and PI0Fast policies are properly integrated with the framework's configuration system.
