#!/usr/bin/env python3
"""
Launch script for the episode visualizer.

This script provides an easy way to run the episode visualizer with appropriate
settings for the IL-Studio repository.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root))

from scripts.visualize.episode_visualizer import EpisodeVisualizer

def find_hdf5_files(data_dir):
    """Find all HDF5 files in the given directory."""
    if os.path.isfile(data_dir) and data_dir.endswith('.hdf5'):
        return [data_dir]
    elif os.path.isdir(data_dir):
        hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
        return sorted(hdf5_files)
    else:
        return []

def main():
    parser = argparse.ArgumentParser(description="IL-Studio Episode Data Visualizer")
    parser.add_argument("data_path", 
                       help="Path to HDF5 episode file or directory containing episodes")
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Host to run server on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, 
                       help="Port to run server on (default: 8050)")
    parser.add_argument("--no-debug", action="store_true", 
                       help="Disable debug mode")
    parser.add_argument("--list-files", action="store_true",
                       help="List all HDF5 files found and exit")
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Path does not exist: {args.data_path}")
        return 1
    
    # Find HDF5 files
    hdf5_files = find_hdf5_files(args.data_path)
    
    if not hdf5_files:
        print(f"Error: No HDF5 files found in: {args.data_path}")
        return 1
    
    print(f"Found {len(hdf5_files)} HDF5 file(s):")
    for i, file_path in enumerate(hdf5_files):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {i+1}. {os.path.basename(file_path)} ({file_size:.1f} MB)")
    
    if args.list_files:
        return 0
    
    print(f"\nStarting visualizer...")
    print(f"URL: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Create and run visualizer
        visualizer = EpisodeVisualizer(args.data_path)
        visualizer.run(host=args.host, port=args.port, debug=not args.no_debug)
    except KeyboardInterrupt:
        print("\nShutting down visualizer...")
        return 0
    except Exception as e:
        print(f"Error: Failed to start visualizer: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

