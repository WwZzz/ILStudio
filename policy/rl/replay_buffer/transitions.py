"""
Transition type definitions for replay buffer.

This module defines the data structures used for storing and batching
RL transitions in ILStudio's replay buffer system.
"""

from typing import TypedDict, Dict, Any, Optional
import torch

from benchmark.base import MetaObs, MetaAction


class RLTransition(TypedDict):
    """Single RL transition in MetaObs/MetaAction format."""
    obs: MetaObs           # Current observation
    action: MetaAction     # Action taken
    reward: float          # Reward received
    next_obs: MetaObs      # Next observation
    done: bool             # Episode terminated
    truncated: bool        # Episode truncated
    info: Optional[Dict[str, Any]]


class BatchTransition(TypedDict):
    """Batched transitions for training."""
    obs: Dict[str, torch.Tensor]       # Batched observations
    action: torch.Tensor               # (B, action_dim) or (B, chunk_size, action_dim)
    reward: torch.Tensor               # (B,)
    next_obs: Dict[str, torch.Tensor]  # Batched next observations
    done: torch.Tensor                 # (B,)
    truncated: torch.Tensor            # (B,)

