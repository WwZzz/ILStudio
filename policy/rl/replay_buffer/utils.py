"""
Utility functions for replay buffer operations.

This module provides utility functions for:
- MetaObs conversion
- Transition to sample conversion
- Data consistency verification
- Environment action sampling
- Normalization and transforms
"""

from typing import Dict, List, Any, Optional, Callable
import torch
import numpy as np
from loguru import logger

from benchmark.base import MetaObs, MetaAction, META_OBS_KEYS
from .transitions import RLTransition


class MetaObsConverter:
    """
    Unified utility class for converting various formats to MetaObs.
    
    This class consolidates MetaObs conversion logic from multiple places:
    - ILReplayBuffer._dict_to_metaobs()
    - ILReplayBuffer._sample_to_metaobs()
    - TransitionConverter._extract_state() (for state extraction)
    
    Provides consistent conversion behavior across the codebase.
    """
    
    @staticmethod
    def from_dict(obs_dict: Dict[str, Any], ctrl_space: str = 'ee') -> MetaObs:
        """
        Convert observation dictionary to MetaObs.
        
        This is a simple conversion that directly maps dictionary keys to MetaObs fields.
        Used for converting environment observations or simple dict observations.
        
        Args:
            obs_dict: Dictionary with observation data (keys matching META_OBS_KEYS)
            ctrl_space: Control space ('ee' or 'joint') - used for state key selection
            
        Returns:
            MetaObs object
        """
        kwargs = {}
        for key in META_OBS_KEYS:
            if key in obs_dict:
                val = obs_dict[key]
                # Convert torch.Tensor to numpy
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                kwargs[key] = val
        return MetaObs(**kwargs)
    
    @staticmethod
    def from_sample(sample: Dict[str, Any], ctrl_space: str = 'ee') -> MetaObs:
        """
        Convert ILStudio dataset sample to MetaObs.
        
        This handles the full conversion from dataset sample format, including:
        - Image format conversion (uint8 normalization)
        - State extraction and control-space specific keys
        - Language instruction extraction
        - Timestamp extraction
        
        Args:
            sample: Dataset sample dictionary (from ILStudio dataset)
            ctrl_space: Control space ('ee' or 'joint') - determines state_ee vs state_joint
            
        Returns:
            MetaObs object
        """
        kwargs = {}
        
        # Image: (N, C, H, W) or (C, H, W)
        if 'image' in sample and sample['image'] is not None:
            img = sample['image']
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            # Convert to uint8 format
            # Check if already in uint8 range by dtype or value range
            if img.dtype == np.uint8:
                kwargs['image'] = img
            else:
                # Check value range - sample a few pixels for efficiency
                sample_size = min(1000, img.size)
                if img.size > 0:
                    sample_indices = np.random.choice(img.size, sample_size, replace=False)
                    img_sample_max = np.max(img.flat[sample_indices])
                else:
                    img_sample_max = 0
                
                if img_sample_max > 1:
                    # Already in [0, 255] range
                    kwargs['image'] = img.astype(np.uint8)
                else:
                    # Normalized to [0, 1], scale to [0, 255]
                    kwargs['image'] = (img * 255).astype(np.uint8)
        
        # State
        if 'state' in sample and sample['state'] is not None:
            state = sample['state']
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            kwargs['state'] = state.astype(np.float32)
            
            # Also set control-space specific state
            if ctrl_space == 'ee':
                kwargs['state_ee'] = state.astype(np.float32)
            elif ctrl_space == 'joint':
                kwargs['state_joint'] = state.astype(np.float32)
        
        # Language instruction
        if 'raw_lang' in sample:
            kwargs['raw_lang'] = sample['raw_lang'] if sample['raw_lang'] else ''
        
        # Timestep
        if 'timestamp' in sample:
            ts = sample['timestamp']
            if isinstance(ts, torch.Tensor):
                ts = ts.item()
            kwargs['timestep'] = int(ts)
        
        return MetaObs(**kwargs)


def transition_to_sample(
    transition: RLTransition, 
    ctrl_space: str = 'ee',
    is_pad: Optional[np.ndarray] = None,
    episode_id: int = 0,
    timestamp: int = 0,
    chunk_size: int = 1,
) -> dict:
    """
    Convert a single RLTransition to dataset sample format.
    
    NOTE: This function is now mainly used for debugging/utility purposes.
    The main sampling flow (sample_processed) builds samples directly from
    storage arrays for efficiency and proper action chunk handling.
    
    For single-step actions (chunk_size=1), this function will expand the
    action to [1, action_dim] format to match dataset conventions.
    
    Args:
        transition: RLTransition dict from replay buffer
        ctrl_space: Control space for state key
        is_pad: Optional padding mask for action
        episode_id: Episode ID for this transition
        timestamp: Timestamp for this transition
        chunk_size: Action chunk size (affects action shape format)
        
    Returns:
        Sample dict in dataset format: {image, state, action, is_pad, raw_lang, ...}
    """
    obs = transition['obs']
    action = transition['action']
    
    sample = {}
    
    # Image
    if obs.image is not None:
        sample['image'] = torch.from_numpy(obs.image)
    
    # State
    if obs.state is not None:
        sample['state'] = torch.from_numpy(obs.state)
    elif obs.state_ee is not None:
        sample['state'] = torch.from_numpy(obs.state_ee)
    elif obs.state_joint is not None:
        sample['state'] = torch.from_numpy(obs.state_joint)
    
    # Action - handle shape conversion for chunk_size=1 case
    if action.action is not None:
        action_tensor = torch.from_numpy(action.action)
        
        # When chunk_size=1, dataset returns action as [1, action_dim] (2D)
        # but replay buffer may store it as [action_dim] (1D) for efficiency
        # We need to match the dataset format
        if chunk_size == 1 and action_tensor.dim() == 1:
            # Expand 1D action to 2D: [action_dim] -> [1, action_dim]
            action_tensor = action_tensor.unsqueeze(0)
        
        sample['action'] = action_tensor
    
    # is_pad - use provided value or create default, handle chunk_size=1 shape
    if is_pad is not None:
        is_pad_tensor = torch.from_numpy(is_pad) if isinstance(is_pad, np.ndarray) else is_pad
        # Handle shape for chunk_size=1: ensure is_pad is 1D array of length 1
        if chunk_size == 1:
            if is_pad_tensor.dim() == 0:
                # Scalar -> [1]
                is_pad_tensor = is_pad_tensor.unsqueeze(0)
            elif is_pad_tensor.dim() == 1 and len(is_pad_tensor) > 1:
                # Multi-element 1D -> take first element and make [1]
                is_pad_tensor = is_pad_tensor[:1]
        sample['is_pad'] = is_pad_tensor
    elif 'action' in sample:
        action_t = sample['action']
        if action_t.dim() == 2:
            sample['is_pad'] = torch.zeros(action_t.shape[0], dtype=torch.bool)
        else:
            # For 1D action (shouldn't happen after expansion), create [1] for chunk_size=1
            sample['is_pad'] = torch.zeros(1, dtype=torch.bool) if chunk_size == 1 else torch.tensor(False)
    
    sample['raw_lang'] = obs.raw_lang if obs.raw_lang else ''
    sample['reasoning'] = ''
    sample['timestamp'] = obs.timestep if obs.timestep is not None else timestamp
    sample['episode_id'] = episode_id
    
    return sample


def random_sample_action_from_env(env, as_metaaction: bool = True, scale: float = 0.1):
    """
    Sample a random action from environment's action bounds.
    
    ILStudio MetaEnv uses min_action/max_action instead of gym action_space.
    MetaEnv.step() expects MetaAction objects, not raw arrays.
    
    Args:
        env: Environment instance
        as_metaaction: If True, wrap action in MetaAction for MetaEnv compatibility
        scale: Scale factor for random actions (0.1 = 10% of full range).
               Small scale helps avoid unstable physics simulation.
        
    Returns:
        MetaAction or np.ndarray depending on as_metaaction flag
    """
    # Sample raw action
    if hasattr(env, 'min_action') and hasattr(env, 'max_action'):
        # ILStudio MetaEnv format
        min_act = np.array(env.min_action)
        max_act = np.array(env.max_action)
        
        # Sample with reduced scale to avoid unstable physics
        # Use center of action space + small random perturbation
        center = (min_act + max_act) / 2.0
        range_half = (max_act - min_act) / 2.0
        
        # Random action within scaled range around center
        action_arr = center + scale * range_half * np.random.uniform(-1, 1, size=min_act.shape)
        action_arr = np.clip(action_arr, min_act, max_act).astype(np.float32)
        
    elif hasattr(env, 'action_space') and hasattr(env.action_space, 'sample'):
        # Standard gym format
        action_arr = env.action_space.sample()
        # Apply scale if it's a Box space
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            center = (env.action_space.low + env.action_space.high) / 2.0
            range_half = (env.action_space.high - env.action_space.low) / 2.0
            action_arr = center + scale * range_half * np.random.uniform(-1, 1, size=env.action_space.shape)
            action_arr = np.clip(action_arr, env.action_space.low, env.action_space.high).astype(np.float32)
    else:
        raise ValueError("Environment must have either min_action/max_action or action_space.sample()")
    
    # Wrap in MetaAction if requested (required for ILStudio MetaEnv)
    if as_metaaction:
        ctrl_space = getattr(env, 'ctrl_space', 'ee')
        ctrl_type = getattr(env, 'ctrl_type', 'delta')
        return MetaAction(
            action=action_arr,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
        )
    
    return action_arr


def apply_normalization_to_sample(
    sample: Dict[str, Any],
    action_normalizer=None,
    state_normalizer=None,
) -> Dict[str, Any]:
    """
    Apply normalization to a sample dict.
    
    This matches the behavior of NormalizedMapDataset.__getitem__ which calls:
    - action_normalizer.normalize(sample['action'], datatype='action')
    - state_normalizer.normalize(sample['state'], datatype='state')
    
    The normalize method preserves the original data type, so we don't force float32.
    
    Args:
        sample: Sample dict with 'state' and 'action' keys
        action_normalizer: Normalizer for actions
        state_normalizer: Normalizer for states
        
    Returns:
        Sample dict with normalized state and action
    """
    if state_normalizer is not None and 'state' in sample:
        state = sample['state']
        # normalize() accepts both torch.Tensor and np.ndarray, and preserves the type
        # IMPORTANT: Must pass datatype='state' to use correct statistics
        normalized_state = state_normalizer.normalize(state, datatype='state')
        # Convert to torch.Tensor if original was torch.Tensor, preserve numpy if original was numpy
        if isinstance(state, torch.Tensor) and not isinstance(normalized_state, torch.Tensor):
            sample['state'] = torch.from_numpy(normalized_state)
        else:
            sample['state'] = normalized_state
    
    if action_normalizer is not None and 'action' in sample:
        action = sample['action']
        # normalize() accepts both torch.Tensor and np.ndarray, and preserves the type
        # IMPORTANT: Must pass datatype='action' to use correct statistics
        normalized_action = action_normalizer.normalize(action, datatype='action')
        # Convert to torch.Tensor if original was torch.Tensor, preserve numpy if original was numpy
        if isinstance(action, torch.Tensor) and not isinstance(normalized_action, torch.Tensor):
            sample['action'] = torch.from_numpy(normalized_action)
        else:
            sample['action'] = normalized_action
    
    return sample


def apply_transforms_to_sample(
    sample: Dict[str, Any],
    transforms: List[Callable],
) -> Dict[str, Any]:
    """
    Apply a list of transforms to a sample dict.
    
    Args:
        sample: Sample dict
        transforms: List of transform functions to apply in order
        
    Returns:
        Transformed sample dict
    """
    for transform in transforms:
        sample = transform(sample)
    return sample

