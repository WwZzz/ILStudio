"""
Test index mapping for filtered LeRobot dataset.
"""

import warnings
warnings.filterwarnings('ignore')

from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset

print("=" * 80)
print("Testing Index Mapping with Filtered Episodes")
print("=" * 80)

# Test with filtered dataset
dataset = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
    episode_filter={"episode_index": [0, 1, 2, 5, 10]}
)

print(f"\nDataset info:")
print(f"  Total episodes: {dataset.total_episodes}")
print(f"  Total frames: {dataset.total_frames}")
print(f"  Episode lengths: {dataset.episode_len}")
print(f"  len(dataset): {len(dataset)}")

# Test accessing data
print(f"\nTesting data access:")
try:
    # Test first frame
    sample = dataset[0]
    print(f"  ✓ dataset[0] successful, episode_id={sample['episode_id']}")
    
    # Test middle frame
    mid_idx = len(dataset) // 2
    sample = dataset[mid_idx]
    print(f"  ✓ dataset[{mid_idx}] successful, episode_id={sample['episode_id']}")
    
    # Test last frame
    last_idx = len(dataset) - 1
    sample = dataset[last_idx]
    print(f"  ✓ dataset[{last_idx}] successful, episode_id={sample['episode_id']}")
    
    print(f"\n✅ All index mappings work correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)


Test index mapping for filtered LeRobot dataset.
"""

import warnings
warnings.filterwarnings('ignore')

from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset

print("=" * 80)
print("Testing Index Mapping with Filtered Episodes")
print("=" * 80)

# Test with filtered dataset
dataset = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
    episode_filter={"episode_index": [0, 1, 2, 5, 10]}
)

print(f"\nDataset info:")
print(f"  Total episodes: {dataset.total_episodes}")
print(f"  Total frames: {dataset.total_frames}")
print(f"  Episode lengths: {dataset.episode_len}")
print(f"  len(dataset): {len(dataset)}")

# Test accessing data
print(f"\nTesting data access:")
try:
    # Test first frame
    sample = dataset[0]
    print(f"  ✓ dataset[0] successful, episode_id={sample['episode_id']}")
    
    # Test middle frame
    mid_idx = len(dataset) // 2
    sample = dataset[mid_idx]
    print(f"  ✓ dataset[{mid_idx}] successful, episode_id={sample['episode_id']}")
    
    # Test last frame
    last_idx = len(dataset) - 1
    sample = dataset[last_idx]
    print(f"  ✓ dataset[{last_idx}] successful, episode_id={sample['episode_id']}")
    
    print(f"\n✅ All index mappings work correctly!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

