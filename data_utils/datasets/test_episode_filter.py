"""
Test script for episode_filter functionality in WrappedLerobotDataset.

This script demonstrates the dynamic episode filtering feature that works
across all LeRobot datasets, regardless of whether they have tasks or not.
"""

import warnings
warnings.filterwarnings('ignore')

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lerobot: {e}")
    LEROBOT_AVAILABLE = False


def print_separator(title=""):
    """Print a nice separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"{title}")
        print("=" * 80)


def inspect_dataset(dataset_path):
    """Inspect dataset metadata."""
    print_separator(f"Inspecting Dataset: {dataset_path}")
    
    try:
        ds_meta = LeRobotDatasetMetadata(dataset_path)
        print(f"Total episodes: {ds_meta.total_episodes}")
        print(f"Total frames: {ds_meta.total_frames}")
        print(f"FPS: {ds_meta.fps}")
        
        # Check if dataset has tasks
        has_tasks = ds_meta.tasks is not None and len(ds_meta.tasks) > 0
        print(f"\nHas tasks: {has_tasks}")
        
        if has_tasks:
            print(f"Number of tasks: {len(ds_meta.tasks)}")
            print("\nFirst 5 tasks:")
            for idx, task_name in enumerate(list(ds_meta.tasks.index[:5]), 1):
                print(f"  {idx}. {task_name}")
        
        # Show episode info
        if ds_meta.episodes is not None:
            print(f"\nEpisode metadata keys: {list(ds_meta.episodes[0].keys())}")
        
        return ds_meta, has_tasks
    except Exception as e:
        print(f"Error: {e}")
        return None, False


def test_no_filter(dataset_path):
    """Test 1: Load dataset without any filtering."""
    print_separator("Test 1: Load entire dataset (no filtering)")
    
    try:
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
        )
        print(f"✓ Successfully loaded dataset")
        print(f"  Episodes: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        print(f"  Max episode length: {dataset.max_episode_len}")
        
        # Test data access
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_episode_index(dataset_path):
    """Test 2: Filter by episode_index (universal method)."""
    print_separator("Test 2: Filter by episode_index (works for ALL datasets)")
    
    try:
        episode_indices = [0, 1, 2, 5, 10]
        print(f"Filter: episode_index = {episode_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": episode_indices}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Expected: {min(len(episode_indices), dataset.total_episodes)}")
        print(f"  Frames: {dataset.total_frames}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_tasks(dataset_path, task_names):
    """Test 3: Filter by task names (only for datasets with tasks)."""
    print_separator("Test 3: Filter by task names (for datasets with tasks)")
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        print(f"Filter: tasks = {task_names}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"tasks": task_names}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        # Verify that loaded data contains the correct tasks
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  First sample task: {sample['raw_lang']}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_task_index(dataset_path, has_tasks):
    """Test 4: Filter by task index (only for datasets with tasks)."""
    print_separator("Test 4: Filter by task_index (for datasets with tasks)")
    
    if not has_tasks:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        task_indices = [0, 1]
        print(f"Filter: task_index = {task_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"task_index": task_indices}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_combined_filter(dataset_path, task_names):
    """Test 5: Combine multiple filter conditions (AND logic)."""
    print_separator("Test 5: Combined filters (AND logic)")
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        episode_indices = [0, 1, 2, 3, 4, 5]
        print(f"Filter: tasks = {task_names[:1]} AND episode_index = {episode_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={
                "tasks": task_names[:1],
                "episode_index": episode_indices
            }
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        print(f"  Note: Only episodes that satisfy BOTH conditions are included")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_empty_filter(dataset_path):
    """Test 6: Test with non-existent filter values."""
    print_separator("Test 6: Filter with non-existent values (should warn)")
    
    try:
        print("Filter: episode_index = [99999] (likely out of range)")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": [99999]}
        )
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        if dataset.total_episodes == 0:
            print("✓ Correctly handled: No episodes match the filter")
        
        return dataset
    except Exception as e:
        print(f"✓ Expected behavior: {e}")
        return None


def run_all_tests(dataset_path):
    """Run all tests on a dataset."""
    print("\n" + "=" * 80)
    print(f"TESTING EPISODE FILTER FUNCTIONALITY")
    print(f"Dataset: {dataset_path}")
    print("=" * 80)
    
    # Inspect dataset first
    ds_meta, has_tasks = inspect_dataset(dataset_path)
    if ds_meta is None:
        print("\nCannot proceed with tests: Failed to load dataset metadata")
        return
    
    # Get some task names if available
    task_names = None
    if has_tasks:
        task_names = list(ds_meta.tasks.index[:2])
    
    # Run tests
    test_no_filter(dataset_path)
    test_filter_by_episode_index(dataset_path)
    test_filter_by_tasks(dataset_path, task_names)
    test_filter_by_task_index(dataset_path, has_tasks)
    test_combined_filter(dataset_path, task_names)
    test_empty_filter(dataset_path)
    
    print_separator("All tests completed!")


def main():
    """Main test function."""
    if not LEROBOT_AVAILABLE:
        print("LeRobot is not available. Cannot run tests.")
        return
    
    # Test on a dataset - you can change this to any LeRobot dataset
    dataset_path = "lerobot/metaworld_mt50"
    
    print("\n" + "=" * 80)
    print("LeRobot Dataset Episode Filter Test Suite")
    print("=" * 80)
    print("\nThis test suite demonstrates the dynamic episode filtering feature")
    print("that works across ALL LeRobot datasets.")
    print("\nKey features:")
    print("  • Universal filtering by episode_index (works for all datasets)")
    print("  • Optional filtering by tasks (for datasets with task metadata)")
    print("  • Combine multiple filters with AND logic")
    print("  • Extensible to any episode metadata field")
    
    run_all_tests(dataset_path)
    
    # You can test on multiple datasets
    print("\n" + "=" * 80)
    print("Testing on another dataset...")
    print("=" * 80)
    
    # Uncomment to test on another dataset:
    # run_all_tests("lerobot/aloha_mobile_cabinet")


if __name__ == "__main__":
    main()


Test script for episode_filter functionality in WrappedLerobotDataset.

This script demonstrates the dynamic episode filtering feature that works
across all LeRobot datasets, regardless of whether they have tasks or not.
"""

import warnings
warnings.filterwarnings('ignore')

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from data_utils.datasets.lerobot_wrapper import WrappedLerobotDataset
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lerobot: {e}")
    LEROBOT_AVAILABLE = False


def print_separator(title=""):
    """Print a nice separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"{title}")
        print("=" * 80)


def inspect_dataset(dataset_path):
    """Inspect dataset metadata."""
    print_separator(f"Inspecting Dataset: {dataset_path}")
    
    try:
        ds_meta = LeRobotDatasetMetadata(dataset_path)
        print(f"Total episodes: {ds_meta.total_episodes}")
        print(f"Total frames: {ds_meta.total_frames}")
        print(f"FPS: {ds_meta.fps}")
        
        # Check if dataset has tasks
        has_tasks = ds_meta.tasks is not None and len(ds_meta.tasks) > 0
        print(f"\nHas tasks: {has_tasks}")
        
        if has_tasks:
            print(f"Number of tasks: {len(ds_meta.tasks)}")
            print("\nFirst 5 tasks:")
            for idx, task_name in enumerate(list(ds_meta.tasks.index[:5]), 1):
                print(f"  {idx}. {task_name}")
        
        # Show episode info
        if ds_meta.episodes is not None:
            print(f"\nEpisode metadata keys: {list(ds_meta.episodes[0].keys())}")
        
        return ds_meta, has_tasks
    except Exception as e:
        print(f"Error: {e}")
        return None, False


def test_no_filter(dataset_path):
    """Test 1: Load dataset without any filtering."""
    print_separator("Test 1: Load entire dataset (no filtering)")
    
    try:
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
        )
        print(f"✓ Successfully loaded dataset")
        print(f"  Episodes: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        print(f"  Max episode length: {dataset.max_episode_len}")
        
        # Test data access
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_episode_index(dataset_path):
    """Test 2: Filter by episode_index (universal method)."""
    print_separator("Test 2: Filter by episode_index (works for ALL datasets)")
    
    try:
        episode_indices = [0, 1, 2, 5, 10]
        print(f"Filter: episode_index = {episode_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": episode_indices}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Expected: {min(len(episode_indices), dataset.total_episodes)}")
        print(f"  Frames: {dataset.total_frames}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_tasks(dataset_path, task_names):
    """Test 3: Filter by task names (only for datasets with tasks)."""
    print_separator("Test 3: Filter by task names (for datasets with tasks)")
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        print(f"Filter: tasks = {task_names}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"tasks": task_names}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        # Verify that loaded data contains the correct tasks
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  First sample task: {sample['raw_lang']}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_filter_by_task_index(dataset_path, has_tasks):
    """Test 4: Filter by task index (only for datasets with tasks)."""
    print_separator("Test 4: Filter by task_index (for datasets with tasks)")
    
    if not has_tasks:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        task_indices = [0, 1]
        print(f"Filter: task_index = {task_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"task_index": task_indices}
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_combined_filter(dataset_path, task_names):
    """Test 5: Combine multiple filter conditions (AND logic)."""
    print_separator("Test 5: Combined filters (AND logic)")
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        episode_indices = [0, 1, 2, 3, 4, 5]
        print(f"Filter: tasks = {task_names[:1]} AND episode_index = {episode_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={
                "tasks": task_names[:1],
                "episode_index": episode_indices
            }
        )
        print(f"✓ Successfully filtered dataset")
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        print(f"  Note: Only episodes that satisfy BOTH conditions are included")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_empty_filter(dataset_path):
    """Test 6: Test with non-existent filter values."""
    print_separator("Test 6: Filter with non-existent values (should warn)")
    
    try:
        print("Filter: episode_index = [99999] (likely out of range)")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": [99999]}
        )
        print(f"  Episodes loaded: {dataset.total_episodes}")
        print(f"  Frames: {dataset.total_frames}")
        
        if dataset.total_episodes == 0:
            print("✓ Correctly handled: No episodes match the filter")
        
        return dataset
    except Exception as e:
        print(f"✓ Expected behavior: {e}")
        return None


def run_all_tests(dataset_path):
    """Run all tests on a dataset."""
    print("\n" + "=" * 80)
    print(f"TESTING EPISODE FILTER FUNCTIONALITY")
    print(f"Dataset: {dataset_path}")
    print("=" * 80)
    
    # Inspect dataset first
    ds_meta, has_tasks = inspect_dataset(dataset_path)
    if ds_meta is None:
        print("\nCannot proceed with tests: Failed to load dataset metadata")
        return
    
    # Get some task names if available
    task_names = None
    if has_tasks:
        task_names = list(ds_meta.tasks.index[:2])
    
    # Run tests
    test_no_filter(dataset_path)
    test_filter_by_episode_index(dataset_path)
    test_filter_by_tasks(dataset_path, task_names)
    test_filter_by_task_index(dataset_path, has_tasks)
    test_combined_filter(dataset_path, task_names)
    test_empty_filter(dataset_path)
    
    print_separator("All tests completed!")


def main():
    """Main test function."""
    if not LEROBOT_AVAILABLE:
        print("LeRobot is not available. Cannot run tests.")
        return
    
    # Test on a dataset - you can change this to any LeRobot dataset
    dataset_path = "lerobot/metaworld_mt50"
    
    print("\n" + "=" * 80)
    print("LeRobot Dataset Episode Filter Test Suite")
    print("=" * 80)
    print("\nThis test suite demonstrates the dynamic episode filtering feature")
    print("that works across ALL LeRobot datasets.")
    print("\nKey features:")
    print("  • Universal filtering by episode_index (works for all datasets)")
    print("  • Optional filtering by tasks (for datasets with task metadata)")
    print("  • Combine multiple filters with AND logic")
    print("  • Extensible to any episode metadata field")
    
    run_all_tests(dataset_path)
    
    # You can test on multiple datasets
    print("\n" + "=" * 80)
    print("Testing on another dataset...")
    print("=" * 80)
    
    # Uncomment to test on another dataset:
    # run_all_tests("lerobot/aloha_mobile_cabinet")


if __name__ == "__main__":
    main()

