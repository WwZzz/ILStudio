#!/usr/bin/env python3
"""
Script to push trained models or datasets to Hugging Face Hub.

Usage examples:
    # Push a model checkpoint to HuggingFace Hub
    python push_to_hub.py --type model --path ckpt/act_example --repo_id username/model-name

    # Push a dataset to HuggingFace Hub
    python push_to_hub.py --type dataset --path data/my_dataset --repo_id username/dataset-name

    # Push with custom commit message
    python push_to_hub.py --type model --path ckpt/act_example --repo_id username/model-name --commit_message "Update model"
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, create_repo


def get_files_to_upload(source_path: Path, ignore_patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Get list of files to upload from source path.
    
    Args:
        source_path: Path to the source directory
        ignore_patterns: List of patterns to ignore (for glob matching)
    
    Returns:
        List of file paths to upload
    """
    files_to_upload = []
    ignore_patterns = ignore_patterns or []
    
    if source_path.is_file():
        return [source_path]
    
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        
        # Filter out directories matching ignore patterns
        dirs_to_remove = []
        for d in dirs:
            dir_path = root_path / d
            rel_path = dir_path.relative_to(source_path)
            
            # Check if directory matches any ignore pattern
            should_ignore = False
            for pattern in ignore_patterns:
                if rel_path.match(pattern) or d.startswith(pattern.replace("*", "")):
                    should_ignore = True
                    break
            
            if should_ignore:
                dirs_to_remove.append(d)
        
        # Remove ignored directories from traversal
        for d in dirs_to_remove:
            dirs.remove(d)
        
        # Add files to upload list
        for file in files:
            file_path = root_path / file
            files_to_upload.append(file_path)
    
    return files_to_upload


def push_model_to_hub(
    ckpt_path: str,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
):
    """
    Push a trained model checkpoint to Hugging Face Hub.
    Ignores checkpoint-* directories (intermediate checkpoints).
    
    Args:
        ckpt_path: Path to the checkpoint directory
        repo_id: Repository ID in format 'username/repo-name'
        commit_message: Optional commit message
        private: Whether to create a private repository
        token: Hugging Face API token (optional, will use cached token if not provided)
    """
    source_path = Path(ckpt_path)
    
    if not source_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}")
    
    print(f"Preparing to push model from: {source_path}")
    print(f"Target repository: {repo_id}")
    
    # Initialize HfApi
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Warning: Could not create/verify repository: {e}")
    
    # Get files to upload, ignoring checkpoint-* directories
    ignore_patterns = ["checkpoint-*"]
    files_to_upload = get_files_to_upload(source_path, ignore_patterns)
    
    if not files_to_upload:
        raise ValueError(f"No files found to upload in {ckpt_path}")
    
    print(f"Found {len(files_to_upload)} files to upload")
    print("Ignoring directories matching: checkpoint-*")
    
    # Upload files
    commit_msg = commit_message or f"Upload model from {source_path.name}"
    
    try:
        # Upload folder to hub
        api.upload_folder(
            folder_path=str(source_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            ignore_patterns=ignore_patterns,
            token=token,
        )
        print(f"✓ Successfully pushed model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Error pushing model: {e}")
        raise


def push_dataset_to_hub(
    dataset_path: str,
    repo_id: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
):
    """
    Push a dataset to Hugging Face Hub.
    Uploads all contents from the specified directory without filtering.
    
    Args:
        dataset_path: Path to the dataset directory
        repo_id: Repository ID in format 'username/repo-name'
        commit_message: Optional commit message
        private: Whether to create a private repository
        token: Hugging Face API token (optional, will use cached token if not provided)
    """
    source_path = Path(dataset_path)
    
    if not source_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Preparing to push dataset from: {source_path}")
    print(f"Target repository: {repo_id}")
    
    # Initialize HfApi
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"Repository {repo_id} is ready")
    except Exception as e:
        print(f"Warning: Could not create/verify repository: {e}")
    
    # Get files to upload (no filtering for datasets)
    files_to_upload = get_files_to_upload(source_path, ignore_patterns=None)
    
    if not files_to_upload:
        raise ValueError(f"No files found to upload in {dataset_path}")
    
    print(f"Found {len(files_to_upload)} files to upload")
    
    # Upload files
    commit_msg = commit_message or f"Upload dataset from {source_path.name}"
    
    try:
        # Upload folder to hub
        api.upload_folder(
            folder_path=str(source_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
            token=token,
        )
        print(f"✓ Successfully pushed dataset to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"✗ Error pushing dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Push trained models or datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Push a model checkpoint
            python push_to_hub.py --type model --path ckpt/act_example --repo_id username/model-name
            
            # Push a dataset
            python push_to_hub.py --type dataset --path data/my_dataset --repo_id username/dataset-name
            
            # Push with custom commit message
            python push_to_hub.py --type model --path ckpt/act_example \\
                --repo_id username/model-name --commit_message "Update model v2"
            
            # Push to a private repository
            python push_to_hub.py --type model --path ckpt/act_example \\
                --repo_id username/model-name --private
        """
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["model", "dataset"],
        help="Type of content to push: 'model' for trained model checkpoints, 'dataset' for datasets"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (for models) or dataset directory (for datasets)"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face Hub in format 'username/repo-name'"
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload (optional)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional, will use cached token if not provided)"
    )
    
    args = parser.parse_args()
    
    # Validate repo_id format
    if "/" not in args.repo_id:
        raise ValueError(
            f"Invalid repo_id format: '{args.repo_id}'. "
            "Expected format: 'username/repo-name'"
        )
    
    # Push to hub based on type
    if args.type == "model":
        push_model_to_hub(
            ckpt_path=args.path,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
        )
    elif args.type == "dataset":
        push_dataset_to_hub(
            dataset_path=args.path,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            private=args.private,
            token=args.token,
        )
    else:
        raise ValueError(f"Unknown type: {args.type}")


if __name__ == "__main__":
    main()

