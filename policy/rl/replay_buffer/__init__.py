"""
Replay Buffer module for ILStudio RL training.

This module provides a modular replay buffer implementation with:
- ILReplayBuffer: Core replay buffer class
- ReplayBufferDataLoader: DataLoader wrapper for training
- Transitions: Type definitions for transitions
- Sampling utilities: Functions for sampling and processing
- Normalization and transforms: On-demand data processing
- Utilities: Conversion and helper functions

All public APIs are exported here for backward compatibility.
"""

# Core classes
from .buffer import ILReplayBuffer
from .loader import ReplayBufferDataLoader
from .rollout_buffer import RolloutReplayBuffer

# Type definitions
from .transitions import RLTransition, BatchTransition

# Utilities
from .utils import MetaObsConverter, transition_to_sample, random_sample_action_from_env

# Sampling functions
from .sampling import (
    build_action_chunk_from_buffer,
    build_sample_from_buffer,
    sample_processed,
    verify_data_consistency,
)

# Normalization and transforms (now in utils)
from .utils import apply_normalization_to_sample, apply_transforms_to_sample

# Public API
__all__ = [
    # Core classes
    'ILReplayBuffer',
    'ReplayBufferDataLoader',
    'RolloutReplayBuffer',
    # Type definitions
    'RLTransition',
    'BatchTransition',
    # Utilities
    'MetaObsConverter',
    'transition_to_sample',
    'random_sample_action_from_env',
    # Sampling functions
    'build_action_chunk_from_buffer',
    'build_sample_from_buffer',
    'sample_processed',
    'verify_data_consistency',
    # Normalization and transforms
    'apply_normalization_to_sample',
    'apply_transforms_to_sample',
]

