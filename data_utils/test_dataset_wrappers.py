"""
Test script for dataset wrappers.

This script demonstrates how the dataset wrappers work with both
map-style and iterable datasets.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
from data_utils.dataset_wrappers import (
    NormalizedMapDataset,
    NormalizedIterableDataset,
    wrap_dataset_with_normalizers
)
import numpy as np


# Mock normalizer for testing
class MockNormalizer:
    """Simple mock normalizer that scales values by a factor."""
    
    def __init__(self, scale_factor=2.0):
        self.scale_factor = scale_factor
    
    def normalize(self, data):
        """Normalize data by multiplying with scale factor."""
        if isinstance(data, np.ndarray):
            return data * self.scale_factor
        elif isinstance(data, torch.Tensor):
            return data * self.scale_factor
        else:
            return data


# Mock map-style dataset
class MockMapDataset(Dataset):
    """Simple map-style dataset for testing."""
    
    def __init__(self, size=10):
        self.size = size
        self.dataset_dir = "mock_dataset_dir"
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'action': np.array([1.0, 2.0, 3.0]),
            'state': np.array([4.0, 5.0, 6.0]),
            'image': np.zeros((64, 64, 3))
        }
    
    def get_dataset_dir(self):
        return self.dataset_dir


# Mock iterable dataset
class MockIterableDataset(IterableDataset):
    """Simple iterable dataset for testing."""
    
    def __init__(self, size=10):
        self.size = size
        self.dataset_dir = "mock_iterable_dataset_dir"
    
    def __iter__(self):
        for i in range(self.size):
            yield {
                'action': np.array([1.0, 2.0, 3.0]),
                'state': np.array([4.0, 5.0, 6.0]),
                'image': np.zeros((64, 64, 3))
            }
    
    def get_dataset_dir(self):
        return self.dataset_dir


def test_map_dataset_wrapper():
    """Test the map-style dataset wrapper."""
    print("Testing Map-style Dataset Wrapper...")
    
    # Create mock dataset and normalizers
    dataset = MockMapDataset(size=5)
    action_normalizers = {
        "mock_dataset_dir": MockNormalizer(scale_factor=2.0)
    }
    state_normalizers = {
        "mock_dataset_dir": MockNormalizer(scale_factor=0.5)
    }
    
    # Wrap the dataset
    wrapped_dataset = NormalizedMapDataset(
        dataset=dataset,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="mock_dataset_dir"
    )
    
    # Test __len__
    assert len(wrapped_dataset) == 5, "Length mismatch"
    print(f"✓ Length: {len(wrapped_dataset)}")
    
    # Test __getitem__
    sample = wrapped_dataset[0]
    print(f"✓ Original action would be: [1.0, 2.0, 3.0]")
    print(f"✓ Normalized action: {sample['action']}")
    print(f"✓ Original state would be: [4.0, 5.0, 6.0]")
    print(f"✓ Normalized state: {sample['state']}")
    
    # Verify normalization
    assert np.allclose(sample['action'], np.array([2.0, 4.0, 6.0])), "Action normalization failed"
    assert np.allclose(sample['state'], np.array([2.0, 2.5, 3.0])), "State normalization failed"
    
    # Test attribute forwarding
    assert wrapped_dataset.get_dataset_dir() == "mock_dataset_dir", "Attribute forwarding failed"
    print(f"✓ Dataset dir from wrapper: {wrapped_dataset.get_dataset_dir()}")
    
    print("✅ Map-style dataset wrapper test passed!\n")


def test_iterable_dataset_wrapper():
    """Test the iterable dataset wrapper."""
    print("Testing Iterable Dataset Wrapper...")
    
    # Create mock dataset and normalizers
    dataset = MockIterableDataset(size=5)
    action_normalizers = {
        "mock_iterable_dataset_dir": MockNormalizer(scale_factor=3.0)
    }
    state_normalizers = {
        "mock_iterable_dataset_dir": MockNormalizer(scale_factor=0.1)
    }
    
    # Wrap the dataset
    wrapped_dataset = NormalizedIterableDataset(
        dataset=dataset,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="mock_iterable_dataset_dir"
    )
    
    # Test iteration
    samples = list(wrapped_dataset)
    assert len(samples) == 5, "Iteration count mismatch"
    print(f"✓ Iterated over {len(samples)} samples")
    
    # Test first sample
    sample = samples[0]
    print(f"✓ Original action would be: [1.0, 2.0, 3.0]")
    print(f"✓ Normalized action: {sample['action']}")
    print(f"✓ Original state would be: [4.0, 5.0, 6.0]")
    print(f"✓ Normalized state: {sample['state']}")
    
    # Verify normalization
    assert np.allclose(sample['action'], np.array([3.0, 6.0, 9.0])), "Action normalization failed"
    assert np.allclose(sample['state'], np.array([0.4, 0.5, 0.6])), "State normalization failed"
    
    # Test attribute forwarding
    assert wrapped_dataset.get_dataset_dir() == "mock_iterable_dataset_dir", "Attribute forwarding failed"
    print(f"✓ Dataset dir from wrapper: {wrapped_dataset.get_dataset_dir()}")
    
    print("✅ Iterable dataset wrapper test passed!\n")


def test_automatic_wrapper():
    """Test the automatic wrapper function."""
    print("Testing Automatic Wrapper Function...")
    
    # Test with map-style dataset
    map_dataset = MockMapDataset(size=3)
    action_normalizers = {"mock_dataset_dir": MockNormalizer(scale_factor=2.0)}
    state_normalizers = {"mock_dataset_dir": MockNormalizer(scale_factor=0.5)}
    
    wrapped_map = wrap_dataset_with_normalizers(
        dataset=map_dataset,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="mock_dataset_dir"
    )
    
    assert isinstance(wrapped_map, NormalizedMapDataset), "Map dataset wrapper type mismatch"
    print("✓ Map-style dataset automatically wrapped correctly")
    
    # Test with iterable dataset
    iterable_dataset = MockIterableDataset(size=3)
    action_normalizers = {"mock_iterable_dataset_dir": MockNormalizer(scale_factor=3.0)}
    state_normalizers = {"mock_iterable_dataset_dir": MockNormalizer(scale_factor=0.1)}
    
    wrapped_iterable = wrap_dataset_with_normalizers(
        dataset=iterable_dataset,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="mock_iterable_dataset_dir"
    )
    
    assert isinstance(wrapped_iterable, NormalizedIterableDataset), "Iterable dataset wrapper type mismatch"
    print("✓ Iterable dataset automatically wrapped correctly")
    
    # Test without normalizers (should return original)
    unwrapped = wrap_dataset_with_normalizers(
        dataset=map_dataset,
        action_normalizers=None,
        state_normalizers=None
    )
    
    assert unwrapped is map_dataset, "Should return original dataset when no normalizers"
    print("✓ Returns original dataset when no normalizers provided")
    
    print("✅ Automatic wrapper function test passed!\n")


def test_duck_typing_custom_datasets():
    """Test that duck typing works with custom dataset classes that don't inherit from Dataset/IterableDataset."""
    print("Testing Duck Typing with Custom Datasets...")
    
    # Custom map-style dataset that doesn't inherit from Dataset
    class CustomMapStyleDataset:
        """Custom dataset with __getitem__ and __len__."""
        def __init__(self):
            self.data = [{'action': np.array([i, i+1, i+2]), 'state': np.array([i*2, i*2+1, i*2+2])} for i in range(5)]
            self.dataset_dir = "custom_map"
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def get_dataset_dir(self):
            return self.dataset_dir
    
    # Custom iterable dataset that doesn't inherit from IterableDataset
    class CustomIterableStyleDataset:
        """Custom dataset with __iter__ only."""
        def __init__(self):
            self.size = 5
            self.dataset_dir = "custom_iterable"
        
        def __iter__(self):
            for i in range(self.size):
                yield {'action': np.array([i, i+1, i+2]), 'state': np.array([i*2, i*2+1, i*2+2])}
        
        def get_dataset_dir(self):
            return self.dataset_dir
    
    # Test custom map-style dataset
    custom_map = CustomMapStyleDataset()
    action_normalizers = {"custom_map": MockNormalizer(scale_factor=2.0)}
    state_normalizers = {"custom_map": MockNormalizer(scale_factor=0.5)}
    
    wrapped_custom_map = wrap_dataset_with_normalizers(
        dataset=custom_map,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="custom_map"
    )
    
    assert isinstance(wrapped_custom_map, NormalizedMapDataset), "Custom map-style dataset should be wrapped as NormalizedMapDataset"
    assert len(wrapped_custom_map) == 5, "Length should be preserved"
    sample = wrapped_custom_map[0]
    assert np.allclose(sample['action'], np.array([0.0, 2.0, 4.0])), "Action normalization failed for custom dataset"
    print("✓ Custom map-style dataset (without inheriting from Dataset) wrapped correctly")
    
    # Test custom iterable dataset
    custom_iterable = CustomIterableStyleDataset()
    action_normalizers = {"custom_iterable": MockNormalizer(scale_factor=3.0)}
    state_normalizers = {"custom_iterable": MockNormalizer(scale_factor=0.1)}
    
    wrapped_custom_iterable = wrap_dataset_with_normalizers(
        dataset=custom_iterable,
        action_normalizers=action_normalizers,
        state_normalizers=state_normalizers,
        dataset_name="custom_iterable"
    )
    
    assert isinstance(wrapped_custom_iterable, NormalizedIterableDataset), "Custom iterable dataset should be wrapped as NormalizedIterableDataset"
    samples = list(wrapped_custom_iterable)
    assert len(samples) == 5, "Should iterate 5 times"
    assert np.allclose(samples[0]['action'], np.array([0.0, 3.0, 6.0])), "Action normalization failed for custom iterable dataset"
    print("✓ Custom iterable dataset (without inheriting from IterableDataset) wrapped correctly")
    
    print("✅ Duck typing with custom datasets test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Wrapper Tests")
    print("=" * 60 + "\n")
    
    test_map_dataset_wrapper()
    test_iterable_dataset_wrapper()
    test_automatic_wrapper()
    test_duck_typing_custom_datasets()
    
    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)

