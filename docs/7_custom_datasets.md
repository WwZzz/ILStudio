# 7. Custom Datasets

IL-Studio is designed to be flexible with data formats. This guide explains how to integrate your own datasets by creating a custom dataset class.

## The `EpisodicDataset` Base Class

The core of data integration is the `EpisodicDataset` class, located in `data_utils/datasets/base.py`. All custom dataset classes should inherit from this base class. It provides essential functionalities:

*   **Episodic Indexing**: Automatically handles mapping a flat sample index to a specific episode and timestep within that episode.
*   **Data Preloading**: Optionally preloads the entire dataset into RAM for faster training.
*   **HDF5 File Discovery**: Automatically finds all `.hdf5` files within a given directory.

## Step 1: Create Your Dataset Class

Create a new Python file in `data_utils/datasets/`, for example, `my_dataset.py`. Inside, define your class inheriting from `EpisodicDataset`.

You **must** implement two key methods:
1.  `get_language_instruction(self, episode_path)`
2.  `load_onestep_from_episode(self, dataset_path, start_ts)`

```python
# In data_utils/datasets/my_dataset.py
from .base import EpisodicDataset
import h5py

class MyCustomDataset(EpisodicDataset):
    def __init__(self, dataset_path_list, camera_names, chunk_size, **kwargs):
        super().__init__(
            dataset_path_list=dataset_path_list,
            camera_names=camera_names,
            chunk_size=chunk_size,
            **kwargs
        )

    def get_language_instruction(self, episode_path):
        """
        Extracts the language instruction from a given episode file.
        This is highly dependent on your HDF5 file's structure.
        """
        with h5py.File(episode_path, 'r') as root:
            # Example: Instruction is stored in the root's attributes
            if 'instruction' in root.attrs:
                return root.attrs['instruction']
            # Example: Instruction is a dataset
            elif 'language_instruction' in root:
                return root['language_instruction'][()].decode('utf-8')
        return "default instruction if not found"

    def load_onestep_from_episode(self, dataset_path, start_ts):
        """
        Loads a single chunk of data starting from a specific timestep.
        """
        with h5py.File(dataset_path, 'r') as root:
            # Load actions for the chunk
            actions = root['actions'][start_ts : start_ts + self.chunk_size]

            # Load states for the chunk
            states = root['observations']['state'][start_ts : start_ts + self.chunk_size]

            # Load images for each camera
            image_dict = {}
            for cam in self.camera_names:
                # Assuming images are stored like /observations/images/cam_name
                image_dict[cam] = root[f'observations/images/{cam}'][start_ts] # Only need first image

        lang = self.get_language_instruction(dataset_path)

        return {
            'action': actions,
            'image': image_dict,
            'state': states,
            'language_instruction': lang,
        }
```

## Step 2: Configure the Dataset

Once you have created your dataset class, reference it in your task configuration file (`configs/task/your_task.yaml`). The framework's dynamic class loader will find and instantiate it.

```yaml
datasets:
  - name: "my_custom_dataset"
    class: "MyCustomDataset"  # The name of your new class
    args:
      dataset_path_list: ['path/to/my/data_directory'] # Directory containing HDF5 files
      camera_names: ['primary', 'wrist']
      chunk_size: 100
      # ... other arguments for the base EpisodicDataset __init__
```

The training script will now use `MyCustomDataset` to load your data. The base class handles the indexing (`__len__`, `_locate_transition`), and your `__getitem__` implementation will be called with the correct episode and timestep.
