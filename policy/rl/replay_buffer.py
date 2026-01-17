"""
Replay Buffer for ILStudio RL training.

This module provides a replay buffer implementation that:
1. Uses MetaObs/MetaAction as the storage format (consistent with ILStudio)
2. Supports action_normalizer and state_normalizer for standardization
3. Can import offline datasets from ILStudio's data loading pipeline
4. Stores online interaction data from environment rollouts

Reference: Aligned with benchmark/base.py MetaObs, MetaAction, and MetaPolicy.
"""

import os
import gc
import threading
import queue
from collections import deque
from contextlib import suppress
from typing import TypedDict, Dict, List, Any, Optional, Union, Callable, Sequence
from dataclasses import dataclass, asdict, fields

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

# Import MetaObs and MetaAction from benchmark.base
from benchmark.base import MetaObs, MetaAction, META_OBS_KEYS, META_ACT_KEYS


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


class ILReplayBuffer:
    """
    Replay buffer for ILStudio RL training using MetaObs/MetaAction format.
    
    Key Features:
    1. Stores observations in MetaObs format (compatible with ILStudio)
    2. Stores actions in MetaAction format with chunk support
    3. Integrates action_normalizer and state_normalizer for standardization
    4. Supports importing from offline datasets
    5. Supports adding online interaction data
    
    Args:
        capacity: Maximum number of transitions to store
        chunk_size: Action chunk size (for chunked action prediction)
        action_normalizer: Normalizer for actions (from data_utils)
        state_normalizer: Normalizer for states (from data_utils)
        ctrl_space: Control space ('ee' or 'joint')
        ctrl_type: Control type ('delta' or 'abs')
        device: Device for sampled tensors
        storage_device: Device for storing data (use 'cpu' to save GPU memory)
    """
    
    def __init__(
        self,
        capacity: int,
        chunk_size: int = 1,
        action_normalizer=None,
        state_normalizer=None,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        device: str = "cuda:0",
        storage_device: str = "cpu",
    ):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.chunk_size = chunk_size
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.device = device
        self.storage_device = storage_device
        
        self.position = 0
        self.size = 0
        self.initialized = False
        
        # Storage will be initialized on first add
        self.obs_storage: Dict[str, np.ndarray] = {}
        self.next_obs_storage: Dict[str, np.ndarray] = {}
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.dones: np.ndarray = None
        self.truncateds: np.ndarray = None
        
        # Track episode boundaries
        self.episode_ends = np.zeros(capacity, dtype=bool)
        
        # Data shapes (set on first add)
        self._obs_shapes: Dict[str, tuple] = {}
        self._obs_dtypes: Dict[str, np.dtype] = {}
        self._action_shape: tuple = None

    def _initialize_storage(
        self,
        obs: MetaObs,
        action: MetaAction,
    ):
        """Initialize storage arrays based on first transition."""
        # Determine observation shapes from MetaObs
        self._obs_shapes = {}
        self._obs_dtypes = {}
        for key in META_OBS_KEYS:
            val = getattr(obs, key, None)
            if val is not None and isinstance(val, np.ndarray):
                self._obs_shapes[key] = val.shape
                self._obs_dtypes[key] = val.dtype
                self.obs_storage[key] = np.empty((self.capacity, *val.shape), dtype=val.dtype)
                self.next_obs_storage[key] = np.empty((self.capacity, *val.shape), dtype=val.dtype)
        
        # Store raw_lang as object array if present
        if obs.raw_lang:
            self.obs_storage['raw_lang'] = np.empty(self.capacity, dtype=object)
            self.next_obs_storage['raw_lang'] = np.empty(self.capacity, dtype=object)
        
        # Determine action shape
        action_arr = action.action
        if action_arr is not None:
            self._action_shape = action_arr.shape
            self.actions = np.empty((self.capacity, *action_arr.shape), dtype=np.float32)
        
        # Scalar arrays
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=bool)
        self.truncateds = np.empty(self.capacity, dtype=bool)
        
        # Store ctrl info
        self.ctrl_spaces = np.empty(self.capacity, dtype=object)
        self.ctrl_types = np.empty(self.capacity, dtype=object)
        
        self.initialized = True
        logger.info(f"Replay buffer initialized with obs_shapes={self._obs_shapes}, action_shape={self._action_shape}")

    def _preallocate_storage(
        self,
        obs_shapes: Dict[str, tuple],
        obs_dtypes: Dict[str, np.dtype],
        action_shape: tuple,
        has_raw_lang: bool = False,
    ):
        """Pre-allocate storage arrays with known shapes (for optimized loading)."""
        self._obs_shapes = obs_shapes
        self._obs_dtypes = obs_dtypes
        self._action_shape = action_shape
        
        for key, shape in obs_shapes.items():
            dtype = obs_dtypes.get(key, np.float32)
            self.obs_storage[key] = np.empty((self.capacity, *shape), dtype=dtype)
            self.next_obs_storage[key] = np.empty((self.capacity, *shape), dtype=dtype)
        
        if has_raw_lang:
            self.obs_storage['raw_lang'] = np.empty(self.capacity, dtype=object)
            self.next_obs_storage['raw_lang'] = np.empty(self.capacity, dtype=object)
        
        if action_shape is not None:
            self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=bool)
        self.truncateds = np.empty(self.capacity, dtype=bool)
        self.ctrl_spaces = np.empty(self.capacity, dtype=object)
        self.ctrl_types = np.empty(self.capacity, dtype=object)
        
        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        obs: MetaObs,
        action: MetaAction,
        reward: float,
        next_obs: MetaObs,
        done: bool,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
        already_normalized: bool = False,
    ):
        """
        Add a single transition to the buffer.
        
        Data is normalized before storage (if normalizers are set and already_normalized=False).
        This ensures all data in the buffer is in normalized form.
        
        Args:
            obs: Current observation (MetaObs)
            action: Action taken (MetaAction)
            reward: Reward received
            next_obs: Next observation (MetaObs)
            done: Whether episode ended
            truncated: Whether episode was truncated
            info: Optional additional info
            already_normalized: If True, skip normalization (data is already normalized)
        """
        # Apply normalization before storage (if normalizers are set)
        if not already_normalized:
            if self.state_normalizer is not None:
                obs = self.state_normalizer.normalize_metaobs(obs, self.ctrl_space)
                next_obs = self.state_normalizer.normalize_metaobs(next_obs, self.ctrl_space)
            if self.action_normalizer is not None:
                action = self.action_normalizer.normalize_metaact(action)
        
        # Initialize storage on first add
        if not self.initialized:
            self._initialize_storage(obs, action)

        # Store observation
        for key in self._obs_shapes:
            val = getattr(obs, key, None)
            if val is not None:
                self.obs_storage[key][self.position] = val
                
            next_val = getattr(next_obs, key, None)
            if next_val is not None:
                self.next_obs_storage[key][self.position] = next_val
        
        # Store raw_lang if present
        if 'raw_lang' in self.obs_storage:
            self.obs_storage['raw_lang'][self.position] = obs.raw_lang
            self.next_obs_storage['raw_lang'][self.position] = next_obs.raw_lang
        
        # Store action (already normalized if normalizer was applied)
        if action.action is not None:
            self.actions[self.position] = action.action
        self.ctrl_spaces[self.position] = action.ctrl_space
        self.ctrl_types[self.position] = action.ctrl_type
        
        # Store scalars
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated
        self.episode_ends[self.position] = done or truncated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_from_env_step(
        self,
        obs: Union[MetaObs, Dict],
        action: Union[MetaAction, np.ndarray],
        reward: float,
        next_obs: Union[MetaObs, Dict],
        done: bool,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a transition from environment step directly.
        
        Data from environment is RAW (not normalized), so normalization will be
        applied in add() before storage.
        
        Accepts both MetaObs/MetaAction and dict/array formats.
        
        Args:
            obs: Current observation (MetaObs or dict) - RAW from env
            action: Action taken (MetaAction or np.ndarray) - RAW from policy output
            reward: Reward received
            next_obs: Next observation (MetaObs or dict) - RAW from env
            done: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional info from environment
        """
        # Convert dict to MetaObs if needed
        if isinstance(obs, dict):
            obs = self._dict_to_metaobs(obs)
        if isinstance(next_obs, dict):
            next_obs = self._dict_to_metaobs(next_obs)
        
        # Convert array to MetaAction if needed
        if isinstance(action, np.ndarray):
            action = MetaAction(
                action=action,
                ctrl_space=self.ctrl_space,
                ctrl_type=self.ctrl_type,
            )
        elif isinstance(action, torch.Tensor):
            action = MetaAction(
                action=action.detach().cpu().numpy(),
                ctrl_space=self.ctrl_space,
                ctrl_type=self.ctrl_type,
            )
        
        # Data from env is raw, needs normalization
        self.add(obs, action, reward, next_obs, done, truncated, info, already_normalized=False)
    
    def _dict_to_metaobs(self, obs_dict: Dict) -> MetaObs:
        """Convert observation dict to MetaObs."""
        kwargs = {}
        for key in META_OBS_KEYS:
            if key in obs_dict:
                val = obs_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                kwargs[key] = val
        return MetaObs(**kwargs)

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a random batch of transitions.
        
        Note: Data in the buffer is already normalized (normalization happens in add()).
        So this method returns normalized data directly without additional processing.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            BatchTransition with tensors on self.device (already normalized)
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)

        # Sample observations (already normalized in storage)
        batch_obs = {}
        batch_next_obs = {}
        
        for key in self._obs_shapes:
            obs_data = self.obs_storage[key][idx].copy()
            next_obs_data = self.next_obs_storage[key][idx].copy()
            
            batch_obs[key] = torch.from_numpy(obs_data).to(self.device)
            batch_next_obs[key] = torch.from_numpy(next_obs_data).to(self.device)
        
        # Handle raw_lang
        if 'raw_lang' in self.obs_storage:
            batch_obs['raw_lang'] = [self.obs_storage['raw_lang'][i] for i in idx]
            batch_next_obs['raw_lang'] = [self.next_obs_storage['raw_lang'][i] for i in idx]
        
        # Sample actions (already normalized in storage)
        batch_actions = self.actions[idx].copy()
        batch_actions = torch.from_numpy(batch_actions).float().to(self.device)
        
        # Sample scalars
        batch_rewards = torch.from_numpy(self.rewards[idx].copy()).float().to(self.device)
        batch_dones = torch.from_numpy(self.dones[idx].copy()).float().to(self.device)
        batch_truncateds = torch.from_numpy(self.truncateds[idx].copy()).float().to(self.device)

        return BatchTransition(
            obs=batch_obs,
            action=batch_actions,
            reward=batch_rewards,
            next_obs=batch_next_obs,
            done=batch_dones,
            truncated=batch_truncateds,
        )

    def sample_as_metaobs(self, batch_size: int) -> List[RLTransition]:
        """
        Sample transitions as a list of RLTransition with MetaObs format.
        
        Useful for algorithms that need the full MetaObs structure.
        Data returned is already normalized (normalization happens in add()).
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of RLTransition dicts (already normalized)
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer.")

        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)
        
        transitions = []
        for i in idx:
            # Reconstruct MetaObs (already normalized in storage)
            obs_kwargs = {key: self.obs_storage[key][i].copy() for key in self._obs_shapes}
            if 'raw_lang' in self.obs_storage:
                obs_kwargs['raw_lang'] = self.obs_storage['raw_lang'][i]
            obs = MetaObs(**obs_kwargs)
            
            next_obs_kwargs = {key: self.next_obs_storage[key][i].copy() for key in self._obs_shapes}
            if 'raw_lang' in self.next_obs_storage:
                next_obs_kwargs['raw_lang'] = self.next_obs_storage['raw_lang'][i]
            next_obs = MetaObs(**next_obs_kwargs)
            
            # Reconstruct MetaAction (already normalized in storage)
            action = MetaAction(
                action=self.actions[i].copy(),
                ctrl_space=self.ctrl_spaces[i],
                ctrl_type=self.ctrl_types[i],
            )
            
            transitions.append(RLTransition(
                obs=obs,
                action=action,
                reward=float(self.rewards[i]),
                next_obs=next_obs,
                done=bool(self.dones[i]),
                truncated=bool(self.truncateds[i]),
                info=None,
            ))
        
        return transitions

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Create an infinite iterator that yields batches of transitions.
        
        Data returned is already normalized (normalization happens in add()).
        
        Args:
            batch_size: Size of batches to sample
            async_prefetch: Use asynchronous prefetching
            queue_size: Number of batches to prefetch
            
        Yields:
            BatchTransition batches (already normalized)
        """
        while True:
            if async_prefetch:
                yield from self._async_iterator(batch_size, queue_size)
            else:
                yield self.sample(batch_size)

    def _async_iterator(self, batch_size: int, queue_size: int):
        """Async iterator with background prefetching."""
        data_queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer():
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    continue
                except Exception:
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True, timeout=1.0)
                except queue.Empty:
                    continue
                except Exception:
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except:
                    pass
            producer_thread.join(timeout=1.0)

    # ============================================================================
    # Import from ILStudio Dataset (Optimized for large datasets)
    # ============================================================================
    
    @classmethod
    def from_ilstudio_dataset(
        cls,
        raw_dataset,
        capacity: Optional[int] = None,
        chunk_size: int = 1,
        action_normalizer=None,
        state_normalizer=None,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        device: str = "cuda:0",
        storage_device: str = "cpu",
        reward_key: str = "reward",
        done_key: str = "done",
        show_progress: bool = True,
        batch_size: int = 1000,
        gc_interval: int = 5000,
    ) -> "ILReplayBuffer":
        """
        Create a replay buffer from an ILStudio dataset.
        
        IMPORTANT: The dataset from load_data() is already normalized and transformed.
        Data is stored directly without additional normalization.
        
        Optimized for large datasets (100k+) with:
        - Pre-allocated memory to avoid dynamic resizing
        - Batch processing to reduce memory peaks
        - Periodic garbage collection
        - Direct array writes (bypasses add() overhead)
        
        Args:
            dataset: ILStudio dataset (from load_data, already normalized/transformed)
            capacity: Buffer capacity. If None, uses dataset length
            chunk_size: Action chunk size
            action_normalizer: Normalizer for actions (stored for add_from_env_step)
            state_normalizer: Normalizer for states (stored for add_from_env_step)
            ctrl_space: Control space ('ee' or 'joint')
            ctrl_type: Control type ('delta' or 'abs')
            device: Device for sampling tensors
            storage_device: Device for storing data
            reward_key: Key for reward in dataset (if available)
            done_key: Key for done flag in dataset (if available)
            show_progress: Show progress bar during loading
            batch_size: Process this many samples before clearing temps (for memory)
            gc_interval: Run garbage collection every N samples
            
        Returns:
            ILReplayBuffer with dataset transitions (already normalized)
        """
        # Unwrap dataset wrappers to get the underlying raw dataset
        # Handles: MapTransformPipeline, NormalizedMapDataset, WrappedDataset, etc.
        dataset_for_data = raw_dataset
        underlying_dataset = raw_dataset
        wrapper_chain = [type(underlying_dataset).__name__]
        while hasattr(underlying_dataset, 'dataset'):
            underlying_dataset = underlying_dataset.dataset
            wrapper_chain.append(type(underlying_dataset).__name__)
        
        logger.info(f"Dataset wrapper chain: {' -> '.join(wrapper_chain)}")
        
        # Try to get ctrl_space/ctrl_type from the underlying dataset if not provided
        if ctrl_space == 'ee' and hasattr(underlying_dataset, 'ctrl_space'):
            ctrl_space = underlying_dataset.ctrl_space
            logger.info(f"  Using ctrl_space from underlying dataset: {ctrl_space}")
        if ctrl_type == 'delta' and hasattr(underlying_dataset, 'ctrl_type'):
            ctrl_type = underlying_dataset.ctrl_type
            logger.info(f"  Using ctrl_type from underlying dataset: {ctrl_type}")
        
        # Use the wrapped dataset for data access (preserves transforms/normalization)
        dataset_len = len(dataset_for_data)
        if capacity is None:
            capacity = dataset_len
        
        if capacity < dataset_len:
            logger.warning(
                f"Buffer capacity ({capacity}) < dataset length ({dataset_len}). "
                f"Only first {capacity} transitions will be stored."
            )
        
        num_to_load = min(capacity, dataset_len)
        
        logger.info(f"Loading ILStudio dataset ({dataset_len} samples) into replay buffer...")
        logger.info(f"  Underlying dataset: {type(underlying_dataset).__name__}")
        logger.info(f"  ctrl_space: {ctrl_space}, ctrl_type: {ctrl_type}")
        logger.info(f"  Capacity: {capacity}, Loading: {num_to_load}")
        logger.info(f"  Batch size: {batch_size}, GC interval: {gc_interval}")
        
        # Step 1: Read first sample to determine shapes
        first_sample = dataset_for_data[0]
        first_obs = cls._sample_to_metaobs(first_sample, ctrl_space)
        
        # Determine action shape
        action_data = first_sample.get('action')
        if action_data is None:
            raise ValueError("Dataset samples must have 'action' key")
        if isinstance(action_data, torch.Tensor):
            action_data = action_data.numpy()
        if action_data.ndim == 2 and chunk_size == 1:
            action_data = action_data[0]
        action_shape = action_data.shape
        
        # Determine obs shapes and dtypes
        obs_shapes = {}
        obs_dtypes = {}
        for key in META_OBS_KEYS:
            val = getattr(first_obs, key, None)
            if val is not None and isinstance(val, np.ndarray):
                obs_shapes[key] = val.shape
                obs_dtypes[key] = val.dtype
        
        has_raw_lang = bool(first_obs.raw_lang)
        
        logger.info(f"  Detected shapes: obs={obs_shapes}, action={action_shape}")
        logger.info(f"  Has raw_lang: {has_raw_lang}")
        
        # Step 2: Create buffer and pre-allocate storage
        replay_buffer = cls(
            capacity=capacity,
            chunk_size=chunk_size,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            device=device,
            storage_device=storage_device,
        )
        
        replay_buffer._preallocate_storage(
            obs_shapes=obs_shapes,
            obs_dtypes=obs_dtypes,
            action_shape=action_shape,
            has_raw_lang=has_raw_lang,
        )
        
        logger.info("  Storage pre-allocated, starting data loading...")
        
        # Step 3: Load data directly into pre-allocated arrays
        # Cache episode_id for boundary detection
        prev_ep_id = None
        
        pbar = tqdm(range(num_to_load), desc="Loading dataset", disable=not show_progress)
        
        for i in pbar:
            sample = dataset_for_data[i]
            
            # Extract observation data directly (avoid MetaObs creation overhead)
            for key in obs_shapes:
                if key == 'image':
                    img = sample.get('image')
                    if img is not None:
                        if isinstance(img, torch.Tensor):
                            img = img.numpy()
                        replay_buffer.obs_storage[key][i] = img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)
                elif key in ['state', 'state_ee', 'state_joint']:
                    state = sample.get('state')
                    if state is not None:
                        if isinstance(state, torch.Tensor):
                            state = state.numpy()
                        replay_buffer.obs_storage[key][i] = state.astype(np.float32)
            
            # Raw lang
            if has_raw_lang:
                replay_buffer.obs_storage['raw_lang'][i] = sample.get('raw_lang', '')
            
            # Action
            action_data = sample.get('action')
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.numpy()
            if action_data.ndim == 2 and chunk_size == 1:
                action_data = action_data[0]
            replay_buffer.actions[i] = action_data.astype(np.float32)
            
            # Reward
            reward = sample.get(reward_key, 0.0)
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            replay_buffer.rewards[i] = float(reward)
            
            # Episode boundary detection
            current_ep = sample.get('episode_id', i)
            is_last_in_episode = False
                
            if i < num_to_load - 1:
                # Peek at next sample's episode_id
                next_sample = dataset_for_data[i + 1]
                next_ep = next_sample.get('episode_id', i + 1)
                if current_ep != next_ep:
                    is_last_in_episode = True
            else:
                is_last_in_episode = True
            
            # Done flag
            done = is_last_in_episode
            if done_key in sample:
                done = bool(sample[done_key])
            
            replay_buffer.dones[i] = done
            replay_buffer.truncateds[i] = False
            replay_buffer.episode_ends[i] = done
            replay_buffer.ctrl_spaces[i] = ctrl_space
            replay_buffer.ctrl_types[i] = ctrl_type
            
            prev_ep_id = current_ep
            
            # Periodic garbage collection
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                gc.collect()
                pbar.set_postfix({'gc': i + 1})
        
        # Step 4: Fill next_obs storage
        # For efficiency, we copy obs_storage and shift
        logger.info("  Filling next_obs storage...")
        
        for key in obs_shapes:
            # next_obs[i] = obs[i+1], except at episode boundaries
            replay_buffer.next_obs_storage[key][:-1] = replay_buffer.obs_storage[key][1:]
            replay_buffer.next_obs_storage[key][-1] = replay_buffer.obs_storage[key][-1]  # last one
        
        if has_raw_lang:
            for i in range(num_to_load - 1):
                replay_buffer.next_obs_storage['raw_lang'][i] = replay_buffer.obs_storage['raw_lang'][i + 1]
            replay_buffer.next_obs_storage['raw_lang'][-1] = replay_buffer.obs_storage['raw_lang'][-1]
        
        # Handle episode boundaries: at episode end, next_obs = obs
        episode_end_indices = np.where(replay_buffer.episode_ends[:num_to_load])[0]
        for key in obs_shapes:
            for end_idx in episode_end_indices:
                replay_buffer.next_obs_storage[key][end_idx] = replay_buffer.obs_storage[key][end_idx]
        
        if has_raw_lang:
            for end_idx in episode_end_indices:
                replay_buffer.next_obs_storage['raw_lang'][end_idx] = replay_buffer.obs_storage['raw_lang'][end_idx]
        
        # Update buffer state
        replay_buffer.size = num_to_load
        replay_buffer.position = num_to_load % capacity
        
        # Final garbage collection
        gc.collect()
        
        stats = replay_buffer.get_statistics()
        logger.info(f"Replay buffer loaded with {stats['size']} transitions.")
        logger.info(f"Statistics: {stats}")
        
        return replay_buffer

    @staticmethod
    def _sample_to_metaobs(sample: Dict[str, Any], ctrl_space: str = 'ee') -> MetaObs:
        """Convert ILStudio dataset sample to MetaObs."""
        kwargs = {}
        
        # Image: (N, C, H, W) or (C, H, W)
        if 'image' in sample and sample['image'] is not None:
            img = sample['image']
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            kwargs['image'] = img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)
        
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

    # ============================================================================
    # Utilities
    # ============================================================================
    
    def clear(self):
        """Clear all data from the buffer."""
        self.position = 0
        self.size = 0
        self.initialized = False
        self.obs_storage = {}
        self.next_obs_storage = {}
        self.actions = None
        self.rewards = None
        self.dones = None
        self.truncateds = None
    
    def _get_obs_at_index(self, idx: int) -> MetaObs:
        """Get observation at specific index (for verification/debugging)."""
        obs_kwargs = {key: self.obs_storage[key][idx].copy() for key in self._obs_shapes}
        if 'raw_lang' in self.obs_storage:
            obs_kwargs['raw_lang'] = self.obs_storage['raw_lang'][idx]
        return MetaObs(**obs_kwargs)
    
    def _get_action_at_index(self, idx: int) -> MetaAction:
        """Get action at specific index (for verification/debugging)."""
        return MetaAction(
            action=self.actions[idx].copy(),
            ctrl_space=self.ctrl_spaces[idx],
            ctrl_type=self.ctrl_types[idx],
        )
    
    def _get_transition_at_index(self, idx: int) -> RLTransition:
        """Get full transition at specific index."""
        obs = self._get_obs_at_index(idx)
        action = self._get_action_at_index(idx)
        next_obs_kwargs = {key: self.next_obs_storage[key][idx].copy() for key in self._obs_shapes}
        if 'raw_lang' in self.next_obs_storage:
            next_obs_kwargs['raw_lang'] = self.next_obs_storage['raw_lang'][idx]
        next_obs = MetaObs(**next_obs_kwargs)
        
        return RLTransition(
            obs=obs,
            action=action,
            reward=float(self.rewards[idx]),
            next_obs=next_obs,
            done=bool(self.dones[idx]),
            truncated=bool(self.truncateds[idx]),
            info=None,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = {
            'size': self.size,
            'capacity': self.capacity,
            'fill_ratio': self.size / self.capacity if self.capacity > 0 else 0,
            'chunk_size': self.chunk_size,
            'ctrl_space': self.ctrl_space,
            'ctrl_type': self.ctrl_type,
            'initialized': self.initialized,
            'has_action_normalizer': self.action_normalizer is not None,
            'has_state_normalizer': self.state_normalizer is not None,
        }
        
        if self.initialized:
            stats['obs_keys'] = list(self._obs_shapes.keys())
            stats['action_shape'] = self._action_shape
            for key, shape in self._obs_shapes.items():
                stats[f'obs_{key}_shape'] = shape
            
            if self.size > 0:
                stats['reward_mean'] = float(self.rewards[:self.size].mean())
                stats['reward_std'] = float(self.rewards[:self.size].std())
                stats['reward_min'] = float(self.rewards[:self.size].min())
                stats['reward_max'] = float(self.rewards[:self.size].max())
                stats['num_episodes'] = int(self.episode_ends[:self.size].sum())
        
        return stats
    
    def save(self, save_path: str):
        """Save buffer to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'capacity': self.capacity,
            'size': self.size,
            'position': self.position,
            'chunk_size': self.chunk_size,
            'ctrl_space': self.ctrl_space,
            'ctrl_type': self.ctrl_type,
            'obs_shapes': self._obs_shapes,
            'action_shape': self._action_shape,
            'obs_storage': self.obs_storage,
            'next_obs_storage': self.next_obs_storage,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'truncateds': self.truncateds,
            'ctrl_spaces': self.ctrl_spaces,
            'ctrl_types': self.ctrl_types,
            'episode_ends': self.episode_ends,
        }
        
        np.savez_compressed(save_path, **save_dict)
        logger.info(f"Buffer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str, device: str = "cuda:0") -> "ILReplayBuffer":
        """Load buffer from disk."""
        data = np.load(load_path, allow_pickle=True)
        
        buffer = cls(
            capacity=int(data['capacity']),
            chunk_size=int(data['chunk_size']),
            ctrl_space=str(data['ctrl_space']),
            ctrl_type=str(data['ctrl_type']),
            device=device,
        )
        
        buffer.size = int(data['size'])
        buffer.position = int(data['position'])
        buffer._obs_shapes = data['obs_shapes'].item()
        buffer._action_shape = tuple(data['action_shape'])
        buffer.obs_storage = data['obs_storage'].item()
        buffer.next_obs_storage = data['next_obs_storage'].item()
        buffer.actions = data['actions']
        buffer.rewards = data['rewards']
        buffer.dones = data['dones']
        buffer.truncateds = data['truncateds']
        buffer.ctrl_spaces = data['ctrl_spaces']
        buffer.ctrl_types = data['ctrl_types']
        buffer.episode_ends = data['episode_ends']
        buffer.initialized = True
        
        logger.info(f"Buffer loaded from {load_path}, size={buffer.size}")
        return buffer


# ============================================================================
# Utility Functions for Replay Buffer
# ============================================================================

def transition_to_sample(transition: RLTransition, ctrl_space: str = 'ee') -> dict:
    """
    Convert a single RLTransition to dataset sample format.
    
    Args:
        transition: RLTransition dict from replay buffer
        ctrl_space: Control space for state key
        
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
    
    # Action
    if action.action is not None:
        sample['action'] = torch.from_numpy(action.action)
    
    # is_pad
    if 'action' in sample:
        action_t = sample['action']
        if action_t.dim() == 2:
            sample['is_pad'] = torch.zeros(action_t.shape[0], dtype=torch.bool)
        else:
            sample['is_pad'] = torch.tensor(False)
    
    sample['raw_lang'] = obs.raw_lang if obs.raw_lang else ''
    sample['reasoning'] = {}
    sample['timestamp'] = obs.timestep if obs.timestep is not None else 0
    sample['episode_id'] = 0
    
    return sample


def sample_processed(
    replay_buffer: ILReplayBuffer, 
    batch_size: int, 
    data_processor=None, 
    device: str = 'cuda:0'
) -> List[Dict[str, Any]]:
    """
    Sample from replay buffer and apply processor.
    
    This is the main function for getting training-ready samples from replay buffer.
    
    Args:
        replay_buffer: The replay buffer (stores normalized data)
        batch_size: Number of samples
        data_processor: Processor to apply to each sample
        device: Device to put tensors on
        
    Returns:
        List of processed samples (each sample is a dict)
    """
    # Sample transitions
    transitions = replay_buffer.sample_as_metaobs(batch_size)
    
    # Convert to dataset sample format
    samples = [transition_to_sample(t, replay_buffer.ctrl_space) for t in transitions]
    
    # Apply processor
    if data_processor is not None:
        samples = [data_processor(s) for s in samples]
    
    # Move tensors to device
    processed = []
    for s in samples:
        s_device = {}
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                s_device[k] = v.to(device)
            else:
                s_device[k] = v
        processed.append(s_device)
    
    return processed


def verify_data_consistency(
    train_data,
    replay_buffer: ILReplayBuffer,
    data_processor=None,
    sample_indices: List[int] = None,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Verify that data from train_data and replay buffer sampling are consistent.
    
    This function compares samples at the same indices from:
    1. train_data (original dataset from load_data)
    2. replay_buffer (sampled and processed)
    
    Args:
        train_data: Dataset from load_data (train.py style)
        replay_buffer: Replay buffer to verify
        data_processor: Processor to apply to replay samples
        sample_indices: Specific indices to compare. If None, uses [0, 1, 2]
        tolerance: Tolerance for float comparison
        
    Returns:
        dict: Verification results with 'passed', 'details', 'mismatches'
    """
    if sample_indices is None:
        sample_indices = [0, 1, 2]
    
    results = {
        'passed': True,
        'details': [],
        'mismatches': [],
    }
    
    logger.info("="*60)
    logger.info("Verifying Data Consistency: train_data vs replay_buffer")
    logger.info("="*60)
    
    # Prefer the passed-in dataset (keeps transforms/normalization).
    # Fallback: unwrap if the dataset is not indexable.
    dataset = train_data
    if not hasattr(dataset, '__getitem__') and hasattr(dataset, 'dataset'):
        while hasattr(dataset, 'dataset') and not hasattr(dataset, '__getitem__'):
            dataset = dataset.dataset
    
    if not hasattr(dataset, '__getitem__'):
        logger.warning("Dataset does not support __getitem__, cannot verify")
        results['passed'] = False
        results['details'].append("Dataset does not support indexing")
        return results
    
    dataset_len = len(dataset) if hasattr(dataset, '__len__') else 0
    logger.info(f"Dataset: {type(dataset).__name__}, len={dataset_len}")
    
    for idx in sample_indices:
        if idx >= len(dataset) or idx >= replay_buffer.size:
            logger.warning(f"Index {idx} out of range, skipping")
            continue
        
        logger.info(f"\n--- Comparing sample at index {idx} ---")
        
        # Get sample from dataset (through processor if available)
        ds_sample = dataset[idx]
        if data_processor is not None:
            ds_sample_processed = data_processor(ds_sample)
        else:
            ds_sample_processed = ds_sample
        
        # Get sample from replay buffer at same index
        rb_transition = replay_buffer._get_transition_at_index(idx)
        rb_sample = transition_to_sample(rb_transition, replay_buffer.ctrl_space)
        if data_processor is not None:
            rb_sample_processed = data_processor(rb_sample)
        else:
            rb_sample_processed = rb_sample
        
        # Compare key fields
        compare_keys = ['image', 'state', 'action', 'is_pad']
        
        for key in compare_keys:
            ds_val = ds_sample_processed.get(key)
            rb_val = rb_sample_processed.get(key)
            
            if ds_val is None and rb_val is None:
                logger.info(f"  {key}: both None ✓")
                continue
            
            if ds_val is None or rb_val is None:
                results['passed'] = False
                msg = f"  {key}: one is None (ds={ds_val is not None}, rb={rb_val is not None}) ✗"
                logger.warning(msg)
                results['mismatches'].append({'index': idx, 'key': key, 'reason': 'one_is_none'})
                continue
            
            # Convert to tensor if needed
            if isinstance(ds_val, np.ndarray):
                ds_val = torch.from_numpy(ds_val)
            if isinstance(rb_val, np.ndarray):
                rb_val = torch.from_numpy(rb_val)
            
            # Compare shapes
            if ds_val.shape != rb_val.shape:
                results['passed'] = False
                msg = f"  {key}: shape mismatch (ds={ds_val.shape}, rb={rb_val.shape}) ✗"
                logger.warning(msg)
                results['mismatches'].append({'index': idx, 'key': key, 'reason': 'shape_mismatch', 
                                              'ds_shape': ds_val.shape, 'rb_shape': rb_val.shape})
                continue
            
            # Compare values
            if ds_val.dtype == torch.bool or rb_val.dtype == torch.bool:
                match = torch.equal(ds_val, rb_val)
            else:
                diff = torch.abs(ds_val.float() - rb_val.float()).max().item()
                match = diff < tolerance
            
            if match:
                logger.info(f"  {key}: shape={ds_val.shape} ✓")
            else:
                results['passed'] = False
                max_diff = torch.abs(ds_val.float() - rb_val.float()).max().item()
                msg = f"  {key}: values differ (max_diff={max_diff:.6f}) ✗"
                logger.warning(msg)
                results['mismatches'].append({'index': idx, 'key': key, 'reason': 'value_mismatch', 
                                              'max_diff': max_diff})
                
                # Show some values for debugging
                logger.info(f"    ds_val[:5]: {ds_val.flatten()[:5]}")
                logger.info(f"    rb_val[:5]: {rb_val.flatten()[:5]}")
    
    logger.info("\n" + "="*60)
    if results['passed']:
        logger.info("✅ Data consistency verification PASSED")
    else:
        logger.warning(f"❌ Data consistency verification FAILED ({len(results['mismatches'])} mismatches)")
    logger.info("="*60)
    
    return results


# ============================================================================
# ReplayBufferDataLoader
# ============================================================================

class ReplayBufferDataLoader:
    """
    Simple DataLoader-like wrapper for replay buffer.
    
    Workflow:
    1. Sample transitions from replay buffer (already normalized)
    2. Convert to dataset sample format
    3. Apply data_processor to each sample
    4. Collate using data_collator
    
    Memory Management:
    - Periodically runs gc.collect() to avoid memory buildup
    - Set gc_interval to control how often GC runs (default: every 100 batches)
    """
    
    def __init__(
        self,
        replay_buffer: ILReplayBuffer,
        batch_size: int,
        num_batches_per_epoch: int = 1000,
        data_processor=None,
        data_collator=None,
        device: str = "cuda:0",
        gc_interval: int = 100,
    ):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.data_processor = data_processor
        self.data_collator = data_collator
        self.device = device
        self.gc_interval = gc_interval
        
    def __len__(self):
        return self.num_batches_per_epoch
    
    def __iter__(self):
        import gc
        for i in range(self.num_batches_per_epoch):
            yield self.sample_batch()
            # Periodic garbage collection to prevent memory buildup
            if self.gc_interval > 0 and (i + 1) % self.gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def sample_batch(self):
        """Sample and process a batch."""
        # Get processed samples
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            self.data_processor,
            device='cpu'  # Keep on CPU, collator/move will handle device
        )
        
        # Collate
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        # Move to device
        return self._to_device(batch, self.device)
    
    def _default_collate(self, samples):
        """Stack tensors, keep lists for non-tensors."""
        if not samples:
            return {}
        batch = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples if key in s]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch
    
    def _to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: self._to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch


# ============================================================================
# Test Functions
# ============================================================================

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
