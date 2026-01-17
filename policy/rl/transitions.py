"""
Transition utilities for RL training in ILStudio.

This module provides utilities for converting between:
1. ILStudio dataset samples → RL transitions
2. Environment observations → RL transitions
3. Policy output → Environment actions

These utilities ensure consistency between offline data and online interaction.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from loguru import logger

# Import RLTransition from replay_buffer
from .replay_buffer import RLTransition


@dataclass
class SimpleTensorTransition:
    """
    Simple tensor-based RL transition format.
    
    Alternative to MetaObs-based RLTransition for simple algorithms.
    Compatible with both ILStudio datasets and environment interactions.
    """
    state: Dict[str, torch.Tensor]       # Current observation
    action: torch.Tensor                  # Action taken
    reward: float                         # Reward received
    next_state: Dict[str, torch.Tensor]  # Next observation
    done: bool                            # Episode terminated
    truncated: bool = False               # Episode truncated (max steps)
    info: Optional[Dict[str, Any]] = None  # Additional information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'truncated': self.truncated,
            'info': self.info,
        }


class TransitionConverter:
    """
    Utility class for converting data between different formats.
    
    Supports:
    1. ILStudio dataset samples → RL transitions
    2. Gym observations → RL state dictionaries
    3. Policy outputs → Environment actions
    """
    
    # Standard ILStudio dataset keys
    ILSTUDIO_IMAGE_KEY = 'image'
    ILSTUDIO_STATE_KEY = 'state'
    ILSTUDIO_ACTION_KEY = 'action'
    ILSTUDIO_LANG_KEY = 'raw_lang'
    ILSTUDIO_EPISODE_KEY = 'episode_id'
    ILSTUDIO_TIMESTAMP_KEY = 'timestamp'
    
    def __init__(
        self,
        state_keys: Optional[List[str]] = None,
        action_dim: Optional[int] = None,
        use_first_action: bool = True,  # For action chunks, use first action
        default_reward: float = 0.0,  # Default reward for IL datasets
    ):
        """
        Initialize the transition converter.
        
        Args:
            state_keys: Keys to extract as state (default: ['image', 'state'])
            action_dim: Action dimension (for validation)
            use_first_action: If True, extract first action from action chunks
            default_reward: Default reward value for datasets without rewards
        """
        self.state_keys = state_keys or ['image', 'state']
        self.action_dim = action_dim
        self.use_first_action = use_first_action
        self.default_reward = default_reward
    
    def ilstudio_sample_to_transition(
        self,
        sample: Dict[str, Any],
        next_sample: Optional[Dict[str, Any]] = None,
        is_last_in_episode: bool = False,
    ) -> RLTransition:
        """
        Convert an ILStudio dataset sample to an RL transition.
        
        Args:
            sample: Current sample from ILStudio dataset
            next_sample: Next sample (for next_state), or None if last
            is_last_in_episode: Whether this is the last sample in episode
            
        Returns:
            RLTransition object
        """
        # Extract state
        state = self._extract_state(sample)
        
        # Extract action
        action = self._extract_action(sample)
        
        # Extract reward (default to 0 for IL datasets)
        reward = sample.get('reward', self.default_reward)
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        
        # Determine done flag
        done = is_last_in_episode
        if 'done' in sample:
            done = bool(sample['done'])
        
        # Extract next_state
        if next_sample is not None and not done:
            next_state = self._extract_state(next_sample)
        else:
            next_state = state.copy()  # Terminal state
        
        # Extract additional info
        info = {}
        if self.ILSTUDIO_LANG_KEY in sample and sample[self.ILSTUDIO_LANG_KEY]:
            info['language'] = sample[self.ILSTUDIO_LANG_KEY]
        if 'is_pad' in sample:
            info['is_pad'] = sample['is_pad']
        
        return RLTransition(
            obs=state,
            action=action,
            reward=reward,
            next_obs=next_state,
            done=done,
            truncated=False,
            info=info if info else None,
        )
    
    def gym_obs_to_state(
        self,
        obs: Union[Dict[str, Any], np.ndarray, torch.Tensor],
        camera_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Gym observation to state dictionary.
        
        Handles various observation formats:
        - Dict observations (most common in robotics)
        - Flat array observations
        - Image-only observations
        
        Args:
            obs: Observation from Gym environment
            camera_names: List of camera names if multiple cameras
            
        Returns:
            State dictionary compatible with ILStudio policies
        """
        state = {}
        
        if isinstance(obs, dict):
            # Handle dict observations
            state = self._convert_dict_obs(obs, camera_names)
        elif isinstance(obs, np.ndarray):
            # Handle flat array - could be state or image
            state = self._convert_array_obs(obs)
        elif isinstance(obs, torch.Tensor):
            state = self._convert_tensor_obs(obs)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        
        return state
    
    def policy_action_to_env_action(
        self,
        action: Union[torch.Tensor, np.ndarray],
        action_space=None,
    ) -> np.ndarray:
        """
        Convert policy output to environment action.
        
        Handles:
        - Action chunks (extracts first action)
        - Tensor to numpy conversion
        - Action space clipping
        
        Args:
            action: Action from policy
            action_space: Optional Gym action space for clipping
            
        Returns:
            Numpy array action for environment
        """
        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Handle batch dimension
        if action.ndim > 1 and action.shape[0] == 1:
            action = action.squeeze(0)
        
        # Handle action chunks (take first action)
        if action.ndim == 2:
            action = action[0]
        
        # Clip to action space if provided
        if action_space is not None:
            action = np.clip(action, action_space.low, action_space.high)
        
        return action.astype(np.float32)
    
    def env_step_to_transition(
        self,
        obs: Dict[str, Any],
        action: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_obs: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
        camera_names: Optional[List[str]] = None,
    ) -> RLTransition:
        """
        Convert environment step output to RL transition.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            terminated: Whether episode ended due to terminal state
            truncated: Whether episode ended due to time limit
            info: Additional info from environment
            camera_names: Camera names if applicable
            
        Returns:
            RLTransition object
        """
        # Convert observations to state dicts
        state = self.gym_obs_to_state(obs, camera_names)
        next_state = self.gym_obs_to_state(next_obs, camera_names)
        
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        return RLTransition(
            obs=state,
            action=action,
            reward=float(reward),
            next_obs=next_state,
            done=terminated,
            truncated=truncated,
            info=info,
        )
    
    def _extract_state(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract state dictionary from sample."""
        state = {}
        
        for key in self.state_keys:
            if key in sample and sample[key] is not None:
                val = sample[key]
                
                # Convert to tensor
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float()
                elif not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                
                # Add batch dimension if needed
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                if val.dim() >= 1 and key == 'image':
                    # Image should be [C, H, W] or [N, C, H, W]
                    if val.dim() == 3:
                        val = val.unsqueeze(0)  # Add batch dim
                elif val.dim() == 1 and key == 'state':
                    val = val.unsqueeze(0)  # Add batch dim
                
                state[key] = val
        
        return state
    
    def _extract_action(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Extract action from sample."""
        action = sample.get(self.ILSTUDIO_ACTION_KEY)
        
        if action is None:
            raise ValueError(f"Sample missing '{self.ILSTUDIO_ACTION_KEY}' key")
        
        # Convert to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        # Handle action chunks - extract first action
        if action.dim() == 2 and self.use_first_action:
            action = action[0]
        
        # Add batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        return action
    
    def _convert_dict_obs(
        self,
        obs: Dict[str, Any],
        camera_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert dict observation to state dictionary."""
        state = {}
        
        # Look for image keys
        image_found = False
        for img_key in ['image', 'rgb', 'pixels', 'agentview_rgb', 'observation.image']:
            if img_key in obs:
                val = self._to_tensor(obs[img_key])
                if val.dim() == 3:  # [H, W, C] or [C, H, W]
                    if val.shape[-1] in [1, 3, 4]:  # [H, W, C]
                        val = val.permute(2, 0, 1)  # -> [C, H, W]
                    val = val.unsqueeze(0)  # -> [1, C, H, W]
                state['image'] = val
                image_found = True
                break
        
        # Handle multiple cameras
        if not image_found and camera_names:
            images = []
            for cam in camera_names:
                if cam in obs:
                    img = self._to_tensor(obs[cam])
                    if img.dim() == 3 and img.shape[-1] in [1, 3, 4]:
                        img = img.permute(2, 0, 1)
                    images.append(img)
            if images:
                state['image'] = torch.stack(images, dim=0).unsqueeze(0)  # [1, N, C, H, W]
        
        # Look for state keys
        for state_key in ['state', 'qpos', 'proprio', 'observation.state', 'robot_state']:
            if state_key in obs:
                val = self._to_tensor(obs[state_key])
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                state['state'] = val
                break
        
        return state
    
    def _convert_array_obs(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convert array observation to state dictionary."""
        val = torch.from_numpy(obs).float()
        
        # Check if it looks like an image
        if val.dim() == 3:
            if val.shape[-1] in [1, 3, 4]:  # [H, W, C]
                val = val.permute(2, 0, 1)
            val = val.unsqueeze(0)
            return {'image': val}
        else:
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return {'state': val}
    
    def _convert_tensor_obs(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert tensor observation to state dictionary."""
        if obs.dim() == 3:
            if obs.shape[-1] in [1, 3, 4]:
                obs = obs.permute(2, 0, 1)
            obs = obs.unsqueeze(0)
            return {'image': obs}
        else:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            return {'state': obs}
    
    def _to_tensor(self, val: Any) -> torch.Tensor:
        """Convert value to tensor."""
        if isinstance(val, torch.Tensor):
            return val.float()
        elif isinstance(val, np.ndarray):
            return torch.from_numpy(val).float()
        else:
            return torch.tensor(val, dtype=torch.float32)


def create_dataset_transition_iterator(
    dataset,
    data_processor: Optional[Callable] = None,
    state_keys: Optional[List[str]] = None,
    batch_size: int = 1,
    shuffle: bool = True,
):
    """
    Create an iterator over transitions from an ILStudio dataset.
    
    This is useful for offline RL training where you want to iterate
    over transitions rather than random sampling.
    
    Args:
        dataset: ILStudio dataset
        data_processor: Optional data processor
        state_keys: State keys to extract
        batch_size: Batch size for iteration
        shuffle: Whether to shuffle data
        
    Yields:
        RLTransition objects (or batches if batch_size > 1)
    """
    converter = TransitionConverter(state_keys=state_keys)
    num_samples = len(dataset)
    
    # Create index order
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    current_batch = []
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if data_processor:
            sample = data_processor(sample)
        
        # Get next sample for next_state
        next_sample = None
        is_last = True
        
        if idx < num_samples - 1:
            next_idx = idx + 1
            next_sample_raw = dataset[next_idx]
            if data_processor:
                next_sample_raw = data_processor(next_sample_raw)
            
            # Check if same episode
            curr_ep = sample.get('episode_id', -1)
            next_ep = next_sample_raw.get('episode_id', -1)
            if curr_ep == next_ep:
                next_sample = next_sample_raw
                is_last = False
        
        transition = converter.ilstudio_sample_to_transition(
            sample=sample,
            next_sample=next_sample,
            is_last_in_episode=is_last,
        )
        
        if batch_size == 1:
            yield transition
        else:
            current_batch.append(transition)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []
    
    # Yield remaining batch
    if current_batch:
        yield current_batch


def batch_transitions(transitions: List[RLTransition]) -> Dict[str, torch.Tensor]:
    """
    Batch a list of transitions into tensors.
    
    Args:
        transitions: List of RLTransition objects
        
    Returns:
        Dictionary with batched tensors
    """
    batch = {
        'obs': {},
        'action': [],
        'reward': [],
        'next_obs': {},
        'done': [],
        'truncated': [],
    }
    
    # Collect all data
    for t in transitions:
        obs = t['obs']
        next_obs = t['next_obs']
        
        # Handle obs - could be MetaObs or dict
        if hasattr(obs, '__dict__'):
            obs_dict = {k: v for k, v in obs.__dict__.items() if v is not None}
        elif isinstance(obs, dict):
            obs_dict = obs
        else:
            obs_dict = {'state': obs}
            
        for key, val in obs_dict.items():
            if val is None:
                continue
            if key not in batch['obs']:
                batch['obs'][key] = []
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            if isinstance(val, torch.Tensor) and val.dim() == 1:
                val = val.unsqueeze(0)
            batch['obs'][key].append(val)
        
        # Handle action
        action = t['action']
        if hasattr(action, 'action'):
            action = action.action  # MetaAction
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        batch['action'].append(action)
        batch['reward'].append(t['reward'])
        
        # Handle next_obs
        if hasattr(next_obs, '__dict__'):
            next_obs_dict = {k: v for k, v in next_obs.__dict__.items() if v is not None}
        elif isinstance(next_obs, dict):
            next_obs_dict = next_obs
        else:
            next_obs_dict = {'state': next_obs}
            
        for key, val in next_obs_dict.items():
            if val is None:
                continue
            if key not in batch['next_obs']:
                batch['next_obs'][key] = []
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            if isinstance(val, torch.Tensor) and val.dim() == 1:
                val = val.unsqueeze(0)
            batch['next_obs'][key].append(val)
        
        batch['done'].append(t['done'])
        batch['truncated'].append(t['truncated'])
    
    # Stack into tensors
    for key in batch['obs']:
        if batch['obs'][key] and isinstance(batch['obs'][key][0], torch.Tensor):
            batch['obs'][key] = torch.cat(batch['obs'][key], dim=0)
    
    if batch['action']:
        batch['action'] = torch.cat(batch['action'], dim=0)
    batch['reward'] = torch.tensor(batch['reward'], dtype=torch.float32)
    
    for key in batch['next_obs']:
        if batch['next_obs'][key] and isinstance(batch['next_obs'][key][0], torch.Tensor):
            batch['next_obs'][key] = torch.cat(batch['next_obs'][key], dim=0)
    
    batch['done'] = torch.tensor(batch['done'], dtype=torch.float32)
    batch['truncated'] = torch.tensor(batch['truncated'], dtype=torch.float32)
    
    return batch

