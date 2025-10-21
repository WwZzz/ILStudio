# Dataset Modules

This directory contains modular dataset implementations for IL-Studio. Each dataset is implemented in its own file for better extensibility and maintainability.

## Structure

```
data_utils/datasets/
├── __init__.py          # Module initialization and exports
├── base.py              # Base EpisodicDataset class
├── aloha_sim.py         # ALOHA simulation datasets
├── aloha_sii.py         # ALOHA SII datasets
├── aloha_sii_v2.py      # ALOHA SII v2 datasets
├── robomimic.py         # RoboMimic benchmark datasets
└── README.md            # This file
```

## Adding New Datasets

To add a new dataset:

1. **Create a new file** (e.g., `my_dataset.py`) in this directory
2. **Import the base class**: `from .base import EpisodicDataset`
3. **Create your dataset class** that inherits from `EpisodicDataset`
4. **Implement required methods**:
   - `get_language_instruction()`: Return task-specific language instruction
   - `load_onestep_from_episode()`: Load single timestep data
   - `load_feat_from_episode()`: Load full episode data
5. **Add to `__init__.py`**: Import and export your new class
6. **Update `data_utils/dataset.py`**: Add import for backward compatibility

## Example

```python
# my_dataset.py
from .base import EpisodicDataset

class MyDataset(EpisodicDataset):
    def get_language_instruction(self):
        return "My custom task instruction"
    
    def load_onestep_from_episode(self, dataset_path, start_ts=None):
        # Your custom loading logic
        return {
            'action': action,
            'image': image_dict,
            'state': state,
            'language_instruction': raw_lang,
            'reasoning': reasoning,
        }
    
    def load_feat_from_episode(self, dataset_path, feats=[]):
        # Your custom loading logic
        return data_dict
```

## Benefits

- **Modularity**: Each dataset is self-contained
- **Extensibility**: Easy to add new datasets without modifying existing code
- **Maintainability**: Clear separation of concerns
- **Backward Compatibility**: Existing code continues to work through `data_utils/dataset.py`

# Gallary

## RLDSWrapper
