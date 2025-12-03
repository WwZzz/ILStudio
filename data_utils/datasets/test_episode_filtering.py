"""
Test script for universal episode filtering functionality in WrappedLerobotDataset.

This script tests the episode_filter parameter which provides flexible filtering
based on episode_index (universal) or other metadata fields like tasks.
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


def inspect_dataset_metadata(dataset_path):
    """Inspect dataset metadata to understand its structure."""
    print("=" * 80)
    print(f"Inspecting dataset: {dataset_path}")
    print("=" * 80)
    
    try:
        ds_meta = LeRobotDatasetMetadata(dataset_path)
        print(f"Total episodes: {ds_meta.total_episodes}")
        print(f"Total frames: {ds_meta.total_frames}")
        print(f"FPS: {ds_meta.fps}")
        
        # Check if dataset has tasks
        if ds_meta.tasks is not None:
            print(f"\n✓ Dataset has tasks: {len(ds_meta.tasks)} tasks")
            print("\nFirst 5 tasks:")
            for idx, task_name in enumerate(list(ds_meta.tasks.index[:5]), 1):
                task_idx = ds_meta.get_task_index(task_name)
                print(f"  {task_idx}. {task_name}")
        else:
            print("\n✗ Dataset does not have tasks metadata")
        
        # Show a few episode examples
        print("\nFirst 3 episodes metadata:")
        for i in range(min(3, len(ds_meta.episodes))):
            ep = ds_meta.episodes[i]
            print(f"\nEpisode {i}:")
            print(f"  Length: {ep.get('length', 'N/A')}")
            print(f"  Tasks: {ep.get('tasks', 'N/A')}")
            # Show other available fields
            other_fields = {k: v for k, v in ep.items() 
                          if k not in ['length', 'tasks', 'dataset_from_index', 'dataset_to_index']}
            if other_fields:
                print(f"  Other fields: {list(other_fields.keys())}")
        
        return ds_meta
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_no_filtering(dataset_path):
    """Test loading dataset without filtering."""
    print("\n" + "=" * 80)
    print("Test 1: Load entire dataset (no filtering)")
    print("=" * 80)
    
    try:
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
        )
        print(f"✓ Total episodes loaded: {dataset.total_episodes}")
        print(f"✓ Total frames loaded: {dataset.total_frames}")
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_filter_by_episode_index(dataset_path):
    """Test filtering by episode_index (universal method)."""
    print("\n" + "=" * 80)
    print("Test 2: Filter by episode_index (universal method)")
    print("=" * 80)
    
    try:
        episode_indices = [0, 1, 2, 5, 10]
        print(f"Filtering episodes: {episode_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": episode_indices}
        )
        print(f"✓ Filtered episodes: {dataset.total_episodes}")
        print(f"✓ Filtered frames: {dataset.total_frames}")
        
        # Verify that data is accessible
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample data keys: {list(sample.keys())}")
        
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_filter_by_tasks(dataset_path, task_names):
    """Test filtering by tasks (if dataset has tasks)."""
    print("\n" + "=" * 80)
    print("Test 3: Filter by tasks (if available)")
    print("=" * 80)
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        print(f"Filtering by tasks: {task_names}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"tasks": task_names}
        )
        print(f"✓ Filtered episodes: {dataset.total_episodes}")
        print(f"✓ Filtered frames: {dataset.total_frames}")
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_filter_by_task_index(dataset_path, task_indices):
    """Test filtering by task_index (if dataset has tasks)."""
    print("\n" + "=" * 80)
    print("Test 4: Filter by task_index (if available)")
    print("=" * 80)
    
    if not task_indices:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        print(f"Filtering by task indices: {task_indices}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"task_index": task_indices}
        )
        print(f"✓ Filtered episodes: {dataset.total_episodes}")
        print(f"✓ Filtered frames: {dataset.total_frames}")
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_invalid_episode_index(dataset_path):
    """Test filtering with invalid episode indices."""
    print("\n" + "=" * 80)
    print("Test 5: Filter with some invalid episode indices (should warn)")
    print("=" * 80)
    
    try:
        # Include some out-of-range indices
        episode_indices = [0, 1, 9999, 10000, 99999]
        print(f"Filtering episodes: {episode_indices} (some invalid)")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={"episode_index": episode_indices}
        )
        print(f"✓ Filtered episodes: {dataset.total_episodes} (invalid ones excluded)")
        print(f"✓ Filtered frames: {dataset.total_frames}")
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def test_combined_filters(dataset_path, task_names):
    """Test combining multiple filter conditions."""
    print("\n" + "=" * 80)
    print("Test 6: Combine multiple filters (AND logic)")
    print("=" * 80)
    
    if not task_names:
        print("⊘ Skipping: Dataset does not have tasks")
        return None
    
    try:
        episode_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print(f"Filtering by episodes: {episode_indices}")
        print(f"AND tasks: {task_names[:1]}")
        
        dataset = WrappedLerobotDataset(
            dataset_path_list=[dataset_path],
            tolerance_s=10.0,
            episode_filter={
                "episode_index": episode_indices,
                "tasks": task_names[:1]
            }
        )
        print(f"✓ Filtered episodes: {dataset.total_episodes}")
        print(f"✓ Filtered frames: {dataset.total_frames}")
        return dataset
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    if not LEROBOT_AVAILABLE:
        print("LeRobot is not available. Cannot run tests.")
        return
    
    # Use a dataset for testing
    # You can change this to any LeRobot dataset you have access to
    dataset_path = "lerobot/metaworld_mt50"
    
    print("=" * 80)
    print("Testing Universal Episode Filtering")
    print("=" * 80)
    print(f"Dataset: {dataset_path}\n")
    
    # Inspect dataset
    ds_meta = inspect_dataset_metadata(dataset_path)
    if ds_meta is None:
        print("\nCannot inspect dataset. Stopping tests.")
        return
    
    # Prepare task names if available
    task_names = None
    task_indices = None
    if ds_meta.tasks is not None and len(ds_meta.tasks) > 0:
        task_names = list(ds_meta.tasks.index[:2])
        task_indices = [0, 1]
    
    # Run tests
    test_no_filtering(dataset_path)
    test_filter_by_episode_index(dataset_path)
    test_filter_by_tasks(dataset_path, task_names)
    test_filter_by_task_index(dataset_path, task_indices)
    test_invalid_episode_index(dataset_path)
    test_combined_filters(dataset_path, task_names)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    print("\nSummary:")
    print("- episode_index filtering: Works for ALL LeRobot datasets")
    print("- tasks filtering: Only works if dataset has tasks metadata")
    print("- task_index filtering: Only works if dataset has tasks metadata")
    print("- Combined filters: Use AND logic to narrow down selection")


if __name__ == "__main__":
    main()

