# RoboTwinDataset Usage Guide

## Overview

`RoboTwinDataset` is an ILStudio dataset loader that directly loads raw RoboTwin HDF5 demonstration data and converts it to ILStudio's standard format **without intermediate processing steps**.

### Key Features

- ✅ **Direct Loading**: Loads raw HDF5 files directly from RoboTwin dataset
- ✅ **Auto-Dimension Detection**: Automatically infers action/state dimensions from data
- ✅ **Efficient Caching**: Episode data cached in memory for fast access
- ✅ **Multi-Source Support**: Load from Hugging Face or local paths (.zip or directories)
- ✅ **Real-Time Processing**: Images decoded on-demand for memory efficiency

## Installation

The dataset requires `h5py`, `cv2`, and `requests` (all included in ILStudio's environment):

```bash
cd /path/to/ILStudio
.venv/bin/pip install h5py opencv-python requests
```

## Usage

### 1. Configuration (YAML)

Add to your training config file (e.g., `configs/task/my_robotwin_task.yaml`):

```yaml
datasets:
  - type: data_utils.datasets.robotwin_dataset.RoboTwinDataset
    name: robotwin_demonstrations
    args:
      # Data source - choose ONE:
      dataset_path: /path/to/data           # Local directory or .zip file
      # OR
      # dataset_name: adjust_bottle/aloha-agilex_clean_50  # Hugging Face path

      image_size: [480, 640]  # [height, width]
      chunk_size: 16          # Frames per sample
      camera_names:
        - head_camera
        - left_camera
        - right_camera
      ctrl_space: qpos        # Joint position control
      ctrl_type: abs          # Absolute control
      preload_data: false     # Set true for small datasets

meta:
  action_dim: 14              # Auto-detected (6 joints + 1 gripper per arm)
  state_dim: 14               # Same as action_dim for qpos control
  action_normalize: zscore
  state_normalize: zscore
```

### 2. Data Source Options

#### Option A: Local Directory
```python
dataset = RoboTwinDataset(
    dataset_path="/home/user/robotwin_data/aloha-agilex_clean_50"
)
```

The directory should contain `episode*.hdf5` files, or a `data/` subdirectory with them.

#### Option B: Local ZIP File
```python
dataset = RoboTwinDataset(
    dataset_path="/home/user/robotwin_data.zip"
)
```

#### Option C: Hugging Face Download
```python
dataset = RoboTwinDataset(
    dataset_name="adjust_bottle/aloha-agilex_clean_50"
)
```

First call will download and cache to `~/.cache/ilstudio/robotwin/`.

### 3. Data Format

Raw RoboTwin HDF5 files contain:

```
/joint_action/
  - left_arm: (T, 6)          # Left arm joint positions
  - left_gripper: (T,)        # Left gripper state (0 or 1)
  - right_arm: (T, 6)         # Right arm joint positions
  - right_gripper: (T,)       # Right gripper state

/observation/
  - head_camera/rgb: (T,)     # Head camera images (JPEG bytes)
  - left_camera/rgb: (T,)     # Left wrist camera images
  - right_camera/rgb: (T,)    # Right wrist camera images
```

### 4. Returned Data Format

Each sample from the dataset is a dictionary:

```python
sample = {
    'state': torch.Tensor([14]),              # Current qpos
    'action': torch.Tensor([16, 14]),         # Action chunk (qpos sequence)
    'image': torch.Tensor([3, 3, 480, 640]), # Images from 3 cameras
    'is_pad': torch.Tensor([16]),             # Padding indicators
    'raw_lang': "",                           # Language (not available in RoboTwin)
    'reasoning': "",
}
```

### 5. Training Integration

```python
from torch.utils.data import DataLoader
from configs.loader import load_config

# Load config
config = load_config('configs/task/my_robotwin_task.yaml')

# Dataset is created automatically
dataset = config.dataset

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Train loop
for batch in dataloader:
    state = batch['state']        # Shape: [batch_size, state_dim]
    actions = batch['action']     # Shape: [batch_size, chunk_size, action_dim]
    images = batch['image']       # Shape: [batch_size, 3, 3, H, W]
    # ... train policy
```

## Advanced Usage

### Custom Cameras

Select specific cameras to reduce memory usage:

```yaml
camera_names:
  - head_camera  # Only load front camera
```

### Statistics Computation

Compute normalization statistics:

```python
dataset = RoboTwinDataset(...)
stats = dataset.get_dataset_statistics()

# Use for normalization
normalizer = {
    'state_mean': stats['state_mean'],
    'state_std': stats['state_std'],
    'action_mean': stats['action_mean'],
    'action_std': stats['action_std'],
}
```

### Memory Efficiency

For very large datasets, use streaming (default):

```yaml
args:
  preload_data: false  # Load episodes on-demand (default)
  chunk_size: 16       # Keep chunk size reasonable
```

For training with multiple workers, each uses its own episode cache.

## Troubleshooting

### Issue: "No episode HDF5 files found"

```
FileNotFoundError: No episode HDF5 files found in: /path/to/data
```

**Solution**: Ensure your data path contains `episode*.hdf5` files:

```bash
ls -la /path/to/data/*.hdf5    # Should show episode files
ls -la /path/to/data/data/     # Or check data/ subdirectory
```

### Issue: "Either 'dataset_path' or 'dataset_name' must be provided"

**Solution**: Add one of these to config:

```yaml
args:
  dataset_path: /path/to/local/data
  # OR
  dataset_name: adjust_bottle/aloha-agilex_clean_50
```

### Issue: Dimension mismatch

```
AssertionError: Expected action shape (chunk_size, 14), got (chunk_size, 16)
```

**Solution**: Dimensions are auto-detected from the HDF5 files. Check:
1. Your data format (some robots have different DOF)
2. Update `meta.action_dim` and `meta.state_dim` in config to match

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Dataset load | 1-2s | Scans for HDF5 files |
| First sample | 0.5-1s | First image decoding |
| Subsequent samples | 0.01-0.05s | From memory cache |
| Batch of 32 | 0.2-0.5s | Parallel processing |

## File Structure

```
data_utils/datasets/
├── robotwin_dataset.py     # Main dataset class
└── ROBOTWIN_USAGE.md       # This file

configs/
└── task/
    └── robotwin_example.yaml  # Example configuration

benchmark/robotwin/
└── README.md              # Integration guide
```

## See Also

- `.cursor/rules/data-rule.mdc` - ILStudio data format specification
- `data_utils/datasets/base.py` - Base dataset class
- `benchmark/robotwin/README.md` - RoboTwin environment integration


