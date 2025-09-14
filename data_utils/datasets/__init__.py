"""
Dataset modules for IL-Studio.

This package contains individual dataset implementations that inherit from the base EpisodicDataset class.
Each dataset is implemented in its own file for better modularity and extensibility.
"""

from .base import EpisodicDataset
from .optimized_base import OptimizedEpisodicDataset, MemoryMappedHDF5, SharedMemoryManager
from .auto_optimizer import DatasetOptimizer, auto_optimize_dataset
from .aloha_sim import AlohaSimDataset
from .aloha_sii import AlohaSIIDataset
from .aloha_sii_v2 import AlohaSIIv2Dataset
from .robomimic import RobomimicDataset
from .koch_dataset import KochDataset

__all__ = [
    'EpisodicDataset',
    'OptimizedEpisodicDataset',
    'MemoryMappedHDF5', 
    'SharedMemoryManager',
    'DatasetOptimizer',
    'auto_optimize_dataset',
    'AlohaSimDataset', 
    'AlohaSIIDataset',
    'AlohaSIIv2Dataset',
    'RobomimicDataset',
    'KochDataset',
]
