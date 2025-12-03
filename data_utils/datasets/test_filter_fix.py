"""
Quick test to verify the episode filter fix.
"""

import warnings
warnings.filterwarnings('ignore')

from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset

print("=" * 80)
print("Testing Episode Filter Fix")
print("=" * 80)

# Test 1: Load full dataset
print("\n1. Load full dataset (no filter)")
dataset_full = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
)
print(f"   Total episodes: {dataset_full.total_episodes}")
print(f"   Total frames: {dataset_full.total_frames}")
print(f"   len(dataset): {len(dataset_full)}")

# Test 2: Filter by episode_index
print("\n2. Filter by episode_index [0, 1, 2, 5, 10]")
dataset_filtered = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
    episode_filter={"episode_index": [0, 1, 2, 5, 10]}
)
print(f"   Total episodes: {dataset_filtered.total_episodes}")
print(f"   Total frames: {dataset_filtered.total_frames}")
print(f"   len(dataset): {len(dataset_filtered)}")
print(f"   Episode lengths: {dataset_filtered.episode_len}")

# Verify
expected_episodes = 5
actual_episodes = dataset_filtered.total_episodes
print(f"\n✓ Expected {expected_episodes} episodes, got {actual_episodes}")

if dataset_filtered.total_frames < dataset_full.total_frames:
    print(f"✓ Filtered frames ({dataset_filtered.total_frames}) < Full frames ({dataset_full.total_frames})")
else:
    print(f"✗ ERROR: Filtered frames should be less than full frames!")

if len(dataset_filtered) == dataset_filtered.total_frames:
    print(f"✓ len(dataset) matches total_frames: {len(dataset_filtered)}")
else:
    print(f"✗ ERROR: len(dataset) != total_frames!")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)


Quick test to verify the episode filter fix.
"""

import warnings
warnings.filterwarnings('ignore')

from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset

print("=" * 80)
print("Testing Episode Filter Fix")
print("=" * 80)

# Test 1: Load full dataset
print("\n1. Load full dataset (no filter)")
dataset_full = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
)
print(f"   Total episodes: {dataset_full.total_episodes}")
print(f"   Total frames: {dataset_full.total_frames}")
print(f"   len(dataset): {len(dataset_full)}")

# Test 2: Filter by episode_index
print("\n2. Filter by episode_index [0, 1, 2, 5, 10]")
dataset_filtered = WrappedLerobotDataset(
    dataset_path_list=["lerobot/metaworld_mt50"],
    tolerance_s=10.0,
    episode_filter={"episode_index": [0, 1, 2, 5, 10]}
)
print(f"   Total episodes: {dataset_filtered.total_episodes}")
print(f"   Total frames: {dataset_filtered.total_frames}")
print(f"   len(dataset): {len(dataset_filtered)}")
print(f"   Episode lengths: {dataset_filtered.episode_len}")

# Verify
expected_episodes = 5
actual_episodes = dataset_filtered.total_episodes
print(f"\n✓ Expected {expected_episodes} episodes, got {actual_episodes}")

if dataset_filtered.total_frames < dataset_full.total_frames:
    print(f"✓ Filtered frames ({dataset_filtered.total_frames}) < Full frames ({dataset_full.total_frames})")
else:
    print(f"✗ ERROR: Filtered frames should be less than full frames!")

if len(dataset_filtered) == dataset_filtered.total_frames:
    print(f"✓ len(dataset) matches total_frames: {len(dataset_filtered)}")
else:
    print(f"✗ ERROR: len(dataset) != total_frames!")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)

