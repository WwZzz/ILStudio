#!/usr/bin/env python3
"""
Dataset Download Script with Retry Mechanism.

This script downloads datasets with automatic retry on rate limit errors.
It uses the same data loading logic as train.py but with robust error handling.

Usage:
    # Download using task config (same as train.py)
    python scripts/download_dataset.py -t lerobot.droid
    
    # Download with custom retry settings
    python scripts/download_dataset.py -t lerobot.droid --max-retries 50 --retry-delay 300
    
    # Download specific dataset directly
    python scripts/download_dataset.py --repo-id cadene/droid_1.0.1 --version v2.1
"""

import os
import sys
import time
import argparse
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars


def download_with_verification(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str = None,
    local_dir: str = None,
    allow_patterns: list = None,
) -> bool:
    """
    Download from HuggingFace with verification that download actually succeeded.
    
    snapshot_download may return local cache path on network errors without raising.
    This function forces a fresh download attempt and verifies success.
    
    Returns:
        True if download succeeded, raises exception otherwise.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    
    # First, verify we can access the remote repository
    # This will raise an exception if there's a network/rate limit issue
    api = HfApi()
    
    try:
        # Try to get repo info - this will fail fast on rate limit
        repo_info = api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )
        logger.debug(f"Successfully accessed repo info for {repo_id}")
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
            raise RuntimeError(f"Rate limit error when accessing {repo_id}: {e}")
        elif "404" in error_str:
            raise RuntimeError(f"Repository not found: {repo_id}")
        else:
            raise RuntimeError(f"Cannot access repository {repo_id}: {e}")
    
    # Now do the actual download
    # Use force_download=False but local_files_only=False to ensure we check remote
    try:
        result_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_files_only=True,  # Force check remote
        )
        logger.debug(f"snapshot_download returned: {result_path}")
        return True
    except Exception as e:
        error_str = str(e)
        # Check if it's a "returning existing local_dir" warning message
        if "returning existing local_dir" in error_str.lower():
            raise RuntimeError(
                f"Download failed - HuggingFace returned cached directory due to network error. "
                f"Original error: {e}"
            )
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets with automatic retry on rate limit errors"
    )
    
    # Task config option (same as train.py)
    parser.add_argument(
        "-t", "--task",
        type=str,
        help="Task config name (e.g., lerobot.droid, lerobot.taco_play)"
    )
    
    # Direct dataset option
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Direct HuggingFace repo ID (e.g., cadene/droid_1.0.1)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2.1",
        choices=["v2.0", "v2.1"],
        help="Dataset version (default: v2.1)"
    )
    
    # Retry settings
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Maximum number of retry attempts (default: 100)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=300,
        help="Delay between retries in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--retry-delay-on-rate-limit",
        type=int,
        default=310,
        help="Delay after rate limit error in seconds (default: 310 = 5min 10s)"
    )
    
    # Other options
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "configs"),
        help="Config directory path"
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        default=True,
        help="Download video files (default: True)"
    )
    parser.add_argument(
        "--no-download-videos",
        action="store_false",
        dest="download_videos",
        help="Skip downloading video files"
    )
    
    return parser.parse_args()


def load_task_config(task_name: str, config_dir: str):
    """Load task configuration from YAML file."""
    import yaml
    
    # Convert task name to path: lerobot.droid -> lerobot/droid.yaml
    task_path = task_name.replace(".", "/") + ".yaml"
    config_path = Path(config_dir) / "task" / task_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def download_with_task_config(task_name: str, config_dir: str, max_retries: int, 
                               retry_delay: int, retry_delay_on_rate_limit: int):
    """Download dataset using task config (same logic as train.py)."""
    # Load task config
    logger.info(f"Loading task config: {task_name}")
    task_config = load_task_config(task_name, config_dir)
    
    datasets_config = task_config.get("datasets", [])
    if not datasets_config:
        raise ValueError(f"No datasets found in task config: {task_name}")
    
    logger.info(f"Found {len(datasets_config)} dataset(s) to download")
    
    for i, ds_config in enumerate(datasets_config):
        ds_type = ds_config.get("type", "")
        ds_name = ds_config.get("name", f"dataset_{i}")
        ds_args = ds_config.get("args", {})
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(datasets_config)}] Downloading: {ds_name}")
        logger.info(f"Type: {ds_type}")
        logger.info(f"Args: {ds_args}")
        logger.info(f"{'='*60}")
        
        download_single_dataset(
            ds_type=ds_type,
            ds_args=ds_args,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_delay_on_rate_limit=retry_delay_on_rate_limit,
        )


def download_single_dataset(ds_type: str, ds_args: dict, max_retries: int,
                            retry_delay: int, retry_delay_on_rate_limit: int):
    """Download a single dataset with retry mechanism."""
    
    attempt = 0
    last_error = None
    
    # Extract repo info from ds_args
    dataset_path_list = ds_args.get("dataset_path_list", [])
    
    while attempt < max_retries:
        attempt += 1
        logger.info(f"\n[Attempt {attempt}/{max_retries}] Starting download...")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: First verify we can access all repositories
            # This catches rate limit errors early before creating dataset
            for repo_id in dataset_path_list:
                logger.info(f"Verifying access to repository: {repo_id}")
                download_with_verification(
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=["meta/*"],  # Just download metadata to verify access
                )
            
            # Step 2: Now create dataset instance (this triggers full download)
            module_path, class_name = ds_type.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            dataset_class = getattr(module, class_name)
            
            logger.info(f"Creating dataset instance...")
            dataset = dataset_class(**ds_args)
            
            # Step 3: Verify dataset is properly loaded
            total_episodes = getattr(dataset, 'total_episodes', 0)
            total_frames = getattr(dataset, 'total_frames', 0)
            
            if total_episodes == 0 or total_frames == 0:
                raise RuntimeError(
                    f"Dataset loaded but appears empty: "
                    f"episodes={total_episodes}, frames={total_frames}"
                )
            
            # Success!
            logger.success(f"\n{'='*60}")
            logger.success(f"Download completed successfully!")
            logger.success(f"Total episodes: {total_episodes}")
            logger.success(f"Total frames: {total_frames}")
            logger.success(f"{'='*60}")
            return True
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a rate limit error or network error
            is_rate_limit = any(x in error_str.lower() for x in [
                "rate limit", "429", "too many requests", "quota",
                "returning existing local_dir", "connection aborted",
                "remote end closed", "remotedisconnected"
            ])
            
            if is_rate_limit:
                delay = retry_delay_on_rate_limit
                logger.warning(f"\n[Rate Limit / Network Error] Cannot access remote repository!")
                logger.warning(f"Error: {error_str[:300]}...")
                logger.warning(f"Waiting {delay} seconds before retry...")
            else:
                delay = retry_delay
                logger.error(f"\n[Error] Download failed!")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {error_str[:500]}")
                logger.warning(f"Waiting {delay} seconds before retry...")
            
            if attempt < max_retries:
                # Show countdown
                for remaining in range(delay, 0, -60):
                    logger.info(f"  Retry in {remaining} seconds...")
                    time.sleep(min(60, remaining))
            else:
                logger.error(f"\nMax retries ({max_retries}) reached. Giving up.")
    
    # All retries failed
    raise RuntimeError(f"Failed to download dataset after {max_retries} attempts. Last error: {last_error}")


def download_direct_repo(repo_id: str, version: str, download_videos: bool,
                         max_retries: int, retry_delay: int, retry_delay_on_rate_limit: int):
    """Download dataset directly by repo ID."""
    
    if version == "v2.1":
        ds_type = "data_utils.datasets.lerobotv21_wrapper.WrappedLerobotV21Dataset"
    else:
        ds_type = "data_utils.datasets.lerobotv20_wrapper.WrappedLerobotV20Dataset"
    
    ds_args = {
        "dataset_path_list": [repo_id],
        "download_videos": download_videos,
    }
    
    logger.info(f"Downloading {repo_id} (version: {version})")
    
    download_single_dataset(
        ds_type=ds_type,
        ds_args=ds_args,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_delay_on_rate_limit=retry_delay_on_rate_limit,
    )


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Dataset Download Script with Retry Mechanism")
    logger.info("=" * 60)
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Retry delay: {args.retry_delay}s")
    logger.info(f"Rate limit delay: {args.retry_delay_on_rate_limit}s")
    
    try:
        if args.task:
            # Download using task config
            download_with_task_config(
                task_name=args.task,
                config_dir=args.config_dir,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                retry_delay_on_rate_limit=args.retry_delay_on_rate_limit,
            )
        elif args.repo_id:
            # Download directly by repo ID
            download_direct_repo(
                repo_id=args.repo_id,
                version=args.version,
                download_videos=args.download_videos,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                retry_delay_on_rate_limit=args.retry_delay_on_rate_limit,
            )
        else:
            logger.error("Please specify either --task or --repo-id")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    logger.success("\nAll downloads completed successfully!")


if __name__ == "__main__":
    main()

