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
from typing import TypedDict, Dict, List, Any, Optional, Union, Callable, Sequence, Tuple
from dataclasses import dataclass, asdict, fields

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

# Import MetaObs and MetaAction from benchmark.base
from benchmark.base import MetaObs, MetaAction, META_OBS_KEYS, META_ACT_KEYS

# Import from new module structure
from .transitions import RLTransition, BatchTransition
from .utils import MetaObsConverter


# ============================================================================
# ILReplayBuffer Core Class
# ============================================================================

class ILReplayBuffer:
    """
    Replay buffer for ILStudio RL training using MetaObs/MetaAction format.
    
    Key Features:
    1. Stores RAW observations in MetaObs format (compatible with ILStudio)
    2. Stores RAW actions in MetaAction format with chunk support
    3. Integrates action_normalizer and state_normalizer for on-demand standardization
    4. Supports transforms pipeline for on-demand data augmentation
    5. Supports importing from offline datasets (stores raw data)
    6. Supports adding online interaction data
    
    Data Flow:
    - Storage: Raw data (from original dataset, bypassing normalizers/transforms)
    - Sampling: Raw data → Normalization → Transforms → Processor → Collator
    
    Args:
        capacity: Maximum number of transitions to store
        chunk_size: Action chunk size (for chunked action prediction)
        action_normalizer: Normalizer for actions (applied on sampling)
        state_normalizer: Normalizer for states (applied on sampling)
        transforms: Transform pipeline to apply on sampling (from data_utils.transform)
        ctrl_space: Control space ('ee' or 'joint')
        ctrl_type: Control type ('delta' or 'abs')
        device: Device for sampled tensors
        storage_device: Device for storing data (use 'cpu' to save GPU memory)
        store_raw: If True, stores raw data; if False, stores normalized data (legacy)
    """
    
    def __init__(
        self,
        capacity: int,
        chunk_size: int = 1,
        action_normalizer=None,
        state_normalizer=None,
        transforms=None,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        device: str = "cuda:0",
        storage_device: str = "cpu",
        store_raw: bool = True,
    ):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.chunk_size = chunk_size
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.transforms = transforms  # Transform pipeline for data augmentation
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.device = device
        self.storage_device = storage_device
        self.store_raw = store_raw  # Whether buffer stores raw (True) or normalized (False) data
        
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
        
        IMPORTANT: By default (store_raw=True), this method stores RAW data without normalization.
        Normalization is applied on-demand during sampling for maximum flexibility.
        
        Args:
            obs: Current observation (MetaObs) - should be RAW data
            action: Action taken (MetaAction) - should be RAW data
            reward: Reward received
            next_obs: Next observation (MetaObs) - should be RAW data
            done: Whether episode ended
            truncated: Whether episode was truncated
            info: Optional additional info
            already_normalized: DEPRECATED - kept for backward compatibility only.
                                When store_raw=True (default), normalization is never applied here.
                                When store_raw=False (legacy), normalization is applied if normalizers exist.
        """
        # Apply normalization before storage ONLY if store_raw=False (legacy mode)
        # In the default mode (store_raw=True), we always store raw data
        if not self.store_raw and not already_normalized:
            # Legacy behavior: normalize before storage
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
        
        Data from environment is RAW (not normalized). With store_raw=True (default),
        this raw data is stored directly without normalization. Normalization will be
        applied on-demand during sampling.
        
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
        
        # Store raw data (normalization happens during sampling if store_raw=True)
        # Pass already_normalized=True to skip normalization in add() when store_raw=True
        # This ensures raw data is always stored in the default mode
        self.add(obs, action, reward, next_obs, done, truncated, info, already_normalized=True)
    
    def _dict_to_metaobs(self, obs_dict: Dict) -> MetaObs:
        """Convert observation dict to MetaObs (uses unified converter)."""
        return MetaObsConverter.from_dict(obs_dict, self.ctrl_space)

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Sample a random batch of transitions.
        
        Note: With store_raw=True (default), this returns RAW data without normalization.
        Use sample_processed() or ReplayBufferDataLoader for normalized and transformed data.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            BatchTransition with tensors on self.device.
            Data is RAW if store_raw=True (default), or normalized if store_raw=False (legacy).
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

    def sample_fast(
        self,
        batch_size: int,
        keys: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> BatchTransition:
        """
        Faster sampling with optional key filtering and no defensive copies.

        This is useful for image-only algorithms (e.g., DrQ) to avoid
        copying unused observation keys and raw_lang.

        Args:
            batch_size: Number of transitions to sample
            keys: Observation keys to include (default: all stored keys)
            device: Target device (default: self.device)

        Returns:
            BatchTransition with tensors on target device
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)
        target_device = device or self.device

        # Limit observation keys
        if keys is None:
            keys = self._obs_shapes.keys()

        batch_obs = {}
        batch_next_obs = {}

        for key in keys:
            if key not in self.obs_storage:
                continue
            obs_data = self.obs_storage[key][idx]
            next_obs_data = self.next_obs_storage[key][idx]
            batch_obs[key] = torch.from_numpy(obs_data).to(target_device)
            batch_next_obs[key] = torch.from_numpy(next_obs_data).to(target_device)

        # Actions and scalars
        batch_actions = torch.from_numpy(self.actions[idx]).float().to(target_device)
        batch_rewards = torch.from_numpy(self.rewards[idx]).float().to(target_device)
        batch_dones = torch.from_numpy(self.dones[idx]).float().to(target_device)
        batch_truncateds = torch.from_numpy(self.truncateds[idx]).float().to(target_device)

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
        With store_raw=True (default), data returned is RAW without normalization.
        Use sample_processed() for normalized and transformed data.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of RLTransition dicts.
            Data is RAW if store_raw=True (default), or normalized if store_raw=False (legacy).
        """
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer.")

        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)
        
        transitions = []
        for i in idx:
            # Reconstruct MetaObs (raw data if store_raw=True, normalized if store_raw=False)
            obs_kwargs = {key: self.obs_storage[key][i].copy() for key in self._obs_shapes}
            if 'raw_lang' in self.obs_storage:
                obs_kwargs['raw_lang'] = self.obs_storage['raw_lang'][i]
            obs = MetaObs(**obs_kwargs)
            
            next_obs_kwargs = {key: self.next_obs_storage[key][i].copy() for key in self._obs_shapes}
            if 'raw_lang' in self.next_obs_storage:
                next_obs_kwargs['raw_lang'] = self.next_obs_storage['raw_lang'][i]
            next_obs = MetaObs(**next_obs_kwargs)
            
            # Reconstruct MetaAction (raw data if store_raw=True, normalized if store_raw=False)
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
        
        With store_raw=True (default), data returned is RAW without normalization.
        Use ReplayBufferDataLoader for normalized and transformed data.
        
        Args:
            batch_size: Size of batches to sample
            async_prefetch: Use asynchronous prefetching
            queue_size: Number of batches to prefetch
            
        Yields:
            BatchTransition batches.
            Data is RAW if store_raw=True (default), or normalized if store_raw=False (legacy).
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
        transforms=None,
        ctrl_space: str = 'ee',
        ctrl_type: str = 'delta',
        device: str = "cuda:0",
        storage_device: str = "cpu",
        reward_key: str = "reward",
        done_key: str = "done",
        show_progress: bool = True,
        batch_size: int = 1000,
        gc_interval: int = 5000,
        store_raw: bool = True,
    ) -> "ILReplayBuffer":
        """
        Create a replay buffer from an ILStudio dataset.
        
        NEW BEHAVIOR (store_raw=True, default):
        - Extracts and stores RAW data from the underlying dataset
        - Bypasses NormalizedMapDataset and MapTransformPipeline wrappers
        - Normalizers and transforms are stored for on-demand application during sampling
        - This provides flexibility for runtime data augmentation and normalization changes
        
        LEGACY BEHAVIOR (store_raw=False):
        - Stores data from the wrapped dataset (already normalized/transformed)
        - Normalizers are only used for add_from_env_step()
        
        Optimized for large datasets (100k+) with:
        - Pre-allocated memory to avoid dynamic resizing
        - Batch processing to reduce memory peaks
        - Periodic garbage collection
        - Direct array writes (bypasses add() overhead)
        
        Args:
            raw_dataset: ILStudio dataset (from load_data, may be wrapped)
            capacity: Buffer capacity. If None, uses dataset length
            chunk_size: Action chunk size
            action_normalizer: Normalizer for actions (applied on sampling if store_raw=True)
            state_normalizer: Normalizer for states (applied on sampling if store_raw=True)
            transforms: Transform pipeline (applied on sampling if store_raw=True)
            ctrl_space: Control space ('ee' or 'joint')
            ctrl_type: Control type ('delta' or 'abs')
            device: Device for sampling tensors
            storage_device: Device for storing data
            reward_key: Key for reward in dataset (if available)
            done_key: Key for done flag in dataset (if available)
            show_progress: Show progress bar during loading
            batch_size: Process this many samples before clearing temps (for memory)
            gc_interval: Run garbage collection every N samples
            store_raw: If True, store raw data and apply normalization/transforms on sampling
            
        Returns:
            ILReplayBuffer with dataset transitions
        """
        # =====================================================================
        # Step 1: Unwrap dataset and extract normalizers/transforms
        # =====================================================================
        current = raw_dataset
        wrapper_chain = [type(current).__name__]
        
        # Track extracted components
        extracted_action_normalizer = action_normalizer
        extracted_state_normalizer = state_normalizer
        extracted_transforms = transforms
        
        # Unwrap and extract normalizers/transforms from wrapper layers
        while hasattr(current, 'dataset'):
            # Extract from NormalizedMapDataset
            if type(current).__name__ == 'NormalizedMapDataset':
                if extracted_action_normalizer is None and hasattr(current, 'action_normalizer'):
                    extracted_action_normalizer = current.action_normalizer
                if extracted_state_normalizer is None and hasattr(current, 'state_normalizer'):
                    extracted_state_normalizer = current.state_normalizer
            
            # Extract from MapTransformPipeline
            if type(current).__name__ == 'MapTransformPipeline':
                if extracted_transforms is None and hasattr(current, 'transforms'):
                    extracted_transforms = current.transforms
            
            current = current.dataset
            wrapper_chain.append(type(current).__name__)
        
        underlying_dataset = current  # The innermost dataset (e.g., EpisodicDataset)
        
        logger.info(f"Dataset wrapper chain: {' -> '.join(wrapper_chain)}")
        logger.info(f"  Underlying dataset: {type(underlying_dataset).__name__}")
        logger.info(f"  Store raw data: {store_raw}")
        logger.info(f"  Extracted action_normalizer: {extracted_action_normalizer is not None}")
        logger.info(f"  Extracted state_normalizer: {extracted_state_normalizer is not None}")
        logger.info(f"  Extracted transforms: {extracted_transforms is not None}")
        
        # =====================================================================
        # Step 2: Choose data source based on store_raw flag
        # =====================================================================
        if store_raw:
            # Use underlying raw dataset (bypasses normalization/transforms)
            dataset_for_data = underlying_dataset
            logger.info("  Loading RAW data from underlying dataset")
        else:
            # Use wrapped dataset (includes normalization/transforms)
            dataset_for_data = raw_dataset
            logger.info("  Loading PROCESSED data from wrapped dataset")
        
        # Try to get ctrl_space/ctrl_type from the underlying dataset
        if ctrl_space == 'ee' and hasattr(underlying_dataset, 'ctrl_space'):
            ctrl_space = underlying_dataset.ctrl_space
            logger.info(f"  Using ctrl_space from underlying dataset: {ctrl_space}")
        if ctrl_type == 'delta' and hasattr(underlying_dataset, 'ctrl_type'):
            ctrl_type = underlying_dataset.ctrl_type
            logger.info(f"  Using ctrl_type from underlying dataset: {ctrl_type}")
        
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
        logger.info(f"  ctrl_space: {ctrl_space}, ctrl_type: {ctrl_type}")
        logger.info(f"  Capacity: {capacity}, Loading: {num_to_load}")
        logger.info(f"  Batch size: {batch_size}, GC interval: {gc_interval}")
        
        # =====================================================================
        # Step 3: Read first sample to determine shapes
        # =====================================================================
        first_sample = dataset_for_data[0]
        first_obs = MetaObsConverter.from_sample(first_sample, ctrl_space)
        
        # Determine action shape - ALWAYS store single-step action (action_dim,)
        # Action chunks are built dynamically during sampling
        action_data = first_sample.get('action')
        if action_data is None:
            raise ValueError("Dataset samples must have 'action' key")
        if isinstance(action_data, torch.Tensor):
            action_data = action_data.numpy()
        
        # Extract single-step action dimension
        if action_data.ndim == 2:
            # Dataset has chunked actions (chunk_size, action_dim) -> take action_dim
            action_dim = action_data.shape[-1]
        else:
            # Dataset has single-step action (action_dim,)
            action_dim = action_data.shape[0]
        
        # Storage shape is always (action_dim,) - single-step
        action_shape = (action_dim,)
        logger.info(f"  Action dim: {action_dim} (storing single-step actions)")
        
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
        
        # =====================================================================
        # Step 4: Create buffer and pre-allocate storage
        # =====================================================================
        replay_buffer = cls(
            capacity=capacity,
            chunk_size=chunk_size,
            action_normalizer=extracted_action_normalizer,
            state_normalizer=extracted_state_normalizer,
            transforms=extracted_transforms,
            ctrl_space=ctrl_space,
            ctrl_type=ctrl_type,
            device=device,
            storage_device=storage_device,
            store_raw=store_raw,
        )
        
        replay_buffer._preallocate_storage(
            obs_shapes=obs_shapes,
            obs_dtypes=obs_dtypes,
            action_shape=action_shape,
            has_raw_lang=has_raw_lang,
        )
        
        logger.info("  Storage pre-allocated, starting data loading...")
        
        # =====================================================================
        # Step 5: Load data directly into pre-allocated arrays
        # Memory optimized: Cache sample to avoid re-reading, use episode_ids array
        # =====================================================================
        
        # Pre-allocate metadata arrays BEFORE the loop to avoid repeated allocation
        replay_buffer.episode_ids = np.zeros(capacity, dtype=np.int64)
        replay_buffer.timestamps = np.zeros(capacity, dtype=np.int64)
        
        # We'll determine is_pads shape from first sample if available
        is_pads_initialized = False
        
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
                        # Memory-optimized: check max value efficiently
                        # For large images, check dtype first to avoid expensive max() call
                        if img.dtype == np.uint8:
                            # Already in uint8 format, direct assignment
                            replay_buffer.obs_storage[key][i] = img
                        else:
                            # Check if values are in [0, 1] range by sampling (faster than full max)
                            # Sample a few pixels to determine range instead of checking entire image
                            sample_size = min(1000, img.size)
                            sample_indices = np.random.choice(img.size, sample_size, replace=False)
                            img_sample_max = np.max(img.flat[sample_indices]) if img.size > 0 else 0
                            
                            if img_sample_max > 1:
                                # Likely already in [0, 255] range, convert to uint8
                                replay_buffer.obs_storage[key][i] = img.astype(np.uint8)
                            else:
                                # Normalized to [0, 1], scale to [0, 255]
                                # Direct assignment to avoid intermediate array if possible
                                replay_buffer.obs_storage[key][i] = (img * 255).astype(np.uint8)
                        # Clear temporary reference immediately
                        del img
                elif key in ['state', 'state_ee', 'state_joint']:
                    state = sample.get('state')
                    if state is not None:
                        if isinstance(state, torch.Tensor):
                            state = state.numpy()
                        # Direct assignment - astype creates new array which is necessary
                        replay_buffer.obs_storage[key][i] = state.astype(np.float32)
                        # Clear temporary reference immediately
                        del state
            
            # Raw lang
            if has_raw_lang:
                replay_buffer.obs_storage['raw_lang'][i] = sample.get('raw_lang', '')
            
            # Action - ALWAYS store single-step action
            # If dataset has chunked actions, take only the first action (current timestep)
            action_data = sample.get('action')
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.numpy()
            
            if action_data.ndim == 2:
                # Chunked action (chunk_size, action_dim) -> take first action
                # Use indexing instead of slicing to avoid creating a view
                action_data = action_data[0].copy() if action_data.shape[0] > 1 else action_data[0]
            
            # Direct assignment - astype creates new array which is necessary
            replay_buffer.actions[i] = action_data.astype(np.float32)
            # Clear temporary reference immediately
            del action_data
            
            # Note: is_pad is NOT stored - it will be computed dynamically during sampling
            # based on episode boundaries and requested chunk_size
            
            # Reward
            reward = sample.get(reward_key, 0.0)
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            replay_buffer.rewards[i] = float(reward)
            
            # Store episode_id (episode boundary detection done AFTER loop)
            current_ep = sample.get('episode_id', i)
            replay_buffer.episode_ids[i] = int(current_ep) if isinstance(current_ep, (int, np.integer)) else i
            
            # Store timestamp
            ts = sample.get('timestamp', i)
            if isinstance(ts, torch.Tensor):
                ts = ts.item()
            replay_buffer.timestamps[i] = int(ts)
            
            # Control info
            replay_buffer.ctrl_spaces[i] = ctrl_space
            replay_buffer.ctrl_types[i] = ctrl_type
            replay_buffer.truncateds[i] = False
            
            # Done flag from sample (will be corrected in post-processing)
            if done_key in sample:
                replay_buffer.dones[i] = bool(sample[done_key])
            else:
                replay_buffer.dones[i] = False
            
            # Clear sample reference to allow GC
            del sample
            
            # Periodic garbage collection
            if gc_interval > 0 and (i + 1) % gc_interval == 0:
                gc.collect()
                pbar.set_postfix({'gc': i + 1})
        
        # =====================================================================
        # Step 5.5: Post-process episode boundaries (AFTER loading all data)
        # This avoids peeking at next sample during the loop
        # =====================================================================
        logger.info("  Detecting episode boundaries...")
        
        # Episode ends where episode_id changes or at the last sample
        episode_ids = replay_buffer.episode_ids[:num_to_load]
        episode_ends = np.zeros(num_to_load, dtype=bool)
        
        # Find where episode_id differs from next sample
        episode_ends[:-1] = episode_ids[:-1] != episode_ids[1:]
        episode_ends[-1] = True  # Last sample is always episode end
        
        # Update dones and episode_ends based on detected boundaries
        # (use OR to preserve any done flags from the dataset)
        replay_buffer.dones[:num_to_load] = np.logical_or(
            replay_buffer.dones[:num_to_load], 
            episode_ends
        )
        replay_buffer.episode_ends[:num_to_load] = episode_ends
        
        num_episodes = np.sum(episode_ends)
        logger.info(f"  Found {num_episodes} episodes in {num_to_load} samples")
        
        # =====================================================================
        # Step 6: Fill next_obs storage
        # =====================================================================
        logger.info("  Filling next_obs storage...")
        
        for key in obs_shapes:
            # next_obs[i] = obs[i+1], except at episode boundaries
            # Memory-optimized: use direct assignment to avoid creating intermediate views
            # Copy in chunks to reduce peak memory usage
            if num_to_load > 1:
                replay_buffer.next_obs_storage[key][:num_to_load-1] = replay_buffer.obs_storage[key][1:num_to_load]
            replay_buffer.next_obs_storage[key][num_to_load-1] = replay_buffer.obs_storage[key][num_to_load-1]  # last one
        
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
        
        # Clear temporary variables to free memory before final GC
        del episode_ids, episode_ends, episode_end_indices
        
        # Final garbage collection - run twice to ensure cleanup
        gc.collect()
        gc.collect()  # Second pass to clean up objects created during first GC
        
        stats = replay_buffer.get_statistics()
        logger.info(f"Replay buffer loaded with {stats['size']} transitions.")
        logger.info(f"Statistics: {stats}")
        
        return replay_buffer

    @staticmethod
    def _sample_to_metaobs(sample: Dict[str, Any], ctrl_space: str = 'ee') -> MetaObs:
        """Convert ILStudio dataset sample to MetaObs (uses unified converter)."""
        return MetaObsConverter.from_sample(sample, ctrl_space)

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
    
    # ============================================================================
    # Episode Retrieval and Video Export
    # ============================================================================
    
    def get_episode_by_id(self, episode_id: int) -> List[Dict[str, Any]]:
        """
        Get all transitions from a specific episode.
        
        Note: This method requires episode_ids and timestamps arrays to be set.
        These are typically set by RolloutReplayBuffer or when loading episodes
        with explicit episode tracking.
        
        Args:
            episode_id: The episode ID to retrieve
            
        Returns:
            List of transition dicts sorted by timestamp, each containing:
            - 'obs': observation dict
            - 'action': action array
            - 'reward': float
            - 'next_obs': next observation dict
            - 'done': bool
            - 'truncated': bool
            - 'timestamp': int
            - 'episode_id': int
        """
        # Check if episode tracking is available
        if not hasattr(self, 'episode_ids') or self.episode_ids is None:
            logger.warning("Episode IDs not available. Use RolloutReplayBuffer for episode tracking.")
            return []
        
        if not hasattr(self, 'timestamps') or self.timestamps is None:
            logger.warning("Timestamps not available. Use RolloutReplayBuffer for episode tracking.")
            return []
        
        # Find all indices for this episode
        mask = self.episode_ids[:self.size] == episode_id
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return []
        
        # Sort by timestamp
        timestamps = self.timestamps[indices]
        sorted_order = np.argsort(timestamps)
        indices = indices[sorted_order]
        
        # Extract transitions
        transitions = []
        for idx in indices:
            obs = {}
            next_obs = {}
            
            for key in self.obs_storage:
                if key == 'raw_lang':
                    obs[key] = self.obs_storage[key][idx]
                    next_obs[key] = self.next_obs_storage[key][idx]
                else:
                    obs[key] = self.obs_storage[key][idx].copy()
                    next_obs[key] = self.next_obs_storage[key][idx].copy()
            
            transition = {
                'obs': obs,
                'action': self.actions[idx].copy(),
                'reward': float(self.rewards[idx]),
                'next_obs': next_obs,
                'done': bool(self.dones[idx]),
                'truncated': bool(self.truncateds[idx]),
                'timestamp': int(self.timestamps[idx]),
                'episode_id': int(self.episode_ids[idx]),
            }
            transitions.append(transition)
        
        return transitions
    
    def save_episode_as_video(
        self,
        episode_id: int,
        output_path: str,
        fps: int = 30,
        camera_idx: int = 0,
    ):
        """
        Save episode observations as a video.
        
        Args:
            episode_id: The episode ID to save as video
            output_path: Path to save the video (e.g., 'episode_0.mp4')
            fps: Video frame rate
            camera_idx: Which camera to use if multiple cameras
        """
        import imageio
        
        episode_data = self.get_episode_by_id(episode_id)
        
        if len(episode_data) == 0:
            logger.warning(f"Empty episode data for episode {episode_id}, skipping video save")
            return
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Collect frames
        frames = []
        for t, transition in enumerate(episode_data):
            obs = transition['obs']
            if 'image' not in obs or obs['image'] is None:
                logger.warning(f"No image in observation at timestep {t}")
                continue
            
            img = obs['image']
            
            # Handle different image shapes
            # (cameras, C, H, W) or (C, H, W)
            if img.ndim == 4:
                # Multiple cameras: select one
                img = img[camera_idx]  # (C, H, W)
            
            if img.ndim == 3:
                # (C, H, W) -> (H, W, C)
                if img.shape[0] in [1, 3, 4]:  # Channels first
                    img = img.transpose(1, 2, 0)
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Handle grayscale
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            
            frames.append(img)
        
        if len(frames) == 0:
            logger.warning("No frames to save")
            return
        
        # Write video
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        logger.info(f"Saved episode {episode_id} video ({len(frames)} frames) to {output_path}")
    
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


