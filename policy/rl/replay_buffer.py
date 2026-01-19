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
        first_obs = cls._sample_to_metaobs(first_sample, ctrl_space)
        
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
            
            # Action - ALWAYS store single-step action
            # If dataset has chunked actions, take only the first action (current timestep)
            action_data = sample.get('action')
            if isinstance(action_data, torch.Tensor):
                action_data = action_data.numpy()
            
            if action_data.ndim == 2:
                # Chunked action (chunk_size, action_dim) -> take first action
                action_data = action_data[0]
            
            replay_buffer.actions[i] = action_data.astype(np.float32)
            
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


# ============================================================================
# Utility Functions for Replay Buffer
# ============================================================================

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


def sample_processed(
    replay_buffer: ILReplayBuffer, 
    batch_size: int, 
    data_processor=None, 
    device: str = 'cuda:0',
    apply_normalization: bool = True,
    apply_transforms: bool = True,
    chunk_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample from replay buffer and apply normalization, transforms, and processor.
    
    NEW SAMPLING STRATEGY:
    1. Randomly sample episode_id
    2. Randomly sample start position within episode
    3. Build action chunk from start position (chunk_size actions)
    4. If actions don't fill chunk, pad with last valid action and set is_pad
    
    Data flow (if store_raw=True):
    1. Sample episodes and start positions
    2. Build action chunks with is_pad
    3. Convert to sample dict format
    4. Apply normalization (if apply_normalization=True and normalizers exist)
    5. Apply transforms (if apply_transforms=True and transforms exist)
    6. Apply data_processor (if provided)
    
    Args:
        replay_buffer: The replay buffer (stores single-step actions)
        batch_size: Number of samples
        data_processor: Processor to apply to each sample
        device: Device to put tensors on
        apply_normalization: Whether to apply normalizers (if buffer stores raw data)
        apply_transforms: Whether to apply transforms (if buffer stores raw data)
        chunk_size: Action chunk size for output (default: replay_buffer.chunk_size)
        
    Returns:
        List of processed samples (each sample is a dict)
    """
    if chunk_size is None:
        chunk_size = replay_buffer.chunk_size
    
    # Build episode index mapping for efficient sampling
    # episode_starts[ep_id] = first index of this episode
    # episode_lengths[ep_id] = length of this episode
    episode_ids = replay_buffer.episode_ids[:replay_buffer.size]
    episode_ends = replay_buffer.episode_ends[:replay_buffer.size]
    
    # Find unique episodes and their boundaries
    unique_episodes = np.unique(episode_ids)
    num_episodes = len(unique_episodes)
    
    # Pre-compute episode boundaries (end indices where episode_ends=True)
    end_indices = np.where(episode_ends)[0]
    
    # Build episode -> (start_idx, end_idx) mapping
    # More efficient: compute start from end_indices
    episode_ranges = {}
    prev_end = -1
    for i, end_idx in enumerate(end_indices):
        start_idx = prev_end + 1
        ep_id = episode_ids[start_idx]
        episode_ranges[ep_id] = (start_idx, end_idx)
        prev_end = end_idx
    
    # Storage references
    obs_storage = replay_buffer.obs_storage
    actions = replay_buffer.actions
    timestamps = replay_buffer.timestamps
    
    # Action dimension
    action_dim = actions.shape[-1] if actions is not None else 0
    
    samples = []
    
    for _ in range(batch_size):
        # Step 1: Randomly sample an episode
        ep_id = np.random.choice(list(episode_ranges.keys()))
        start_idx, end_idx = episode_ranges[ep_id]
        ep_length = end_idx - start_idx + 1
        
        # Step 2: Randomly sample start position within episode
        frame_offset = np.random.randint(0, ep_length)
        global_idx = start_idx + frame_offset
        
        # Step 3: Build action chunk with padding (like lerobotv21_wrapper.py)
        # Calculate how many valid actions we can get
        remaining_in_episode = ep_length - frame_offset
        valid_count = min(chunk_size, remaining_in_episode)
        
        if valid_count > 0:
            # Get valid actions
            valid_end = global_idx + valid_count
            valid_actions = actions[global_idx:valid_end].copy()
            
            if valid_count < chunk_size:
                # Need padding: repeat last valid action
                pad_count = chunk_size - valid_count
                last_action = valid_actions[-1:] if len(valid_actions) > 0 else actions[end_idx:end_idx+1]
                padding = np.repeat(last_action, pad_count, axis=0)
                action_chunk = np.concatenate([valid_actions, padding], axis=0)
                is_pad = np.array([False] * valid_count + [True] * pad_count, dtype=bool)
            else:
                action_chunk = valid_actions
                is_pad = np.array([False] * chunk_size, dtype=bool)
        else:
            # All padding (shouldn't happen normally)
            last_action = actions[end_idx:end_idx+1]
            action_chunk = np.repeat(last_action, chunk_size, axis=0)
            is_pad = np.array([True] * chunk_size, dtype=bool)
        
        # Build sample dict
        sample = {}
        
        # Image
        if 'image' in obs_storage:
            sample['image'] = torch.from_numpy(obs_storage['image'][global_idx])
        
        # State
        state_key = None
        for k in ['state', 'state_ee', 'state_joint']:
            if k in obs_storage:
                state_key = k
                break
        if state_key is not None:
            sample['state'] = torch.from_numpy(obs_storage[state_key][global_idx].copy())
        
        # Action chunk
        sample['action'] = torch.from_numpy(action_chunk.astype(np.float32))
        
        # is_pad
        sample['is_pad'] = torch.from_numpy(is_pad)
        
        # Metadata
        if 'raw_lang' in obs_storage:
            sample['raw_lang'] = obs_storage['raw_lang'][global_idx] or ''
        else:
            sample['raw_lang'] = ''
        
        sample['reasoning'] = ''
        sample['episode_id'] = int(ep_id)
        sample['timestamp'] = int(timestamps[global_idx]) if timestamps is not None else frame_offset
        
        samples.append(sample)
    
    # Apply normalization if buffer stores raw data
    if replay_buffer.store_raw and apply_normalization:
        for i, sample in enumerate(samples):
            samples[i] = apply_normalization_to_sample(
                sample,
                action_normalizer=replay_buffer.action_normalizer,
                state_normalizer=replay_buffer.state_normalizer,
            )
    
    # Apply transforms if buffer stores raw data
    if replay_buffer.store_raw and apply_transforms and replay_buffer.transforms is not None:
        transforms = replay_buffer.transforms
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        for i, sample in enumerate(samples):
            samples[i] = apply_transforms_to_sample(sample, transforms)
    
    # Apply processor
    if data_processor is not None:
        samples = [data_processor(s) for s in samples]
    
    # Move tensors to device
    processed = []
    for s in samples:
        s_device = {}
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                if v.device.type != device.split(':')[0] or (device.startswith('cuda:') and hasattr(v.device, 'index') and v.device.index != int(device.split(':')[1])):
                    s_device[k] = v.to(device, non_blocking=True)
                else:
                    s_device[k] = v
            else:
                s_device[k] = v
        processed.append(s_device)
    
    del samples
    return processed


def verify_data_consistency(
    train_data,
    replay_buffer: ILReplayBuffer,
    data_processor=None,
    sample_indices: List[int] = None,
    tolerance: float = 1e-5,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify that data from train_data and replay buffer sampling are consistent.
    
    NEW: Replay buffer stores single-step actions, so we need to build action chunks
    dynamically for comparison with the dataset.
    
    This function compares samples at the same indices from:
    1. train_data (original dataset from load_data, may have action chunks)
    2. replay_buffer (single-step actions, chunks built dynamically)
    
    Args:
        train_data: Dataset from load_data (train.py style)
        replay_buffer: Replay buffer to verify
        data_processor: Processor to apply to replay samples
        sample_indices: Specific indices to compare. If None, uses [0, 1, 2]
        tolerance: Tolerance for float comparison
        chunk_size: Chunk size for action chunks (default: replay_buffer.chunk_size)
        
    Returns:
        dict: Verification results with 'passed', 'details', 'mismatches'
    """
    if sample_indices is None:
        sample_indices = [0, 1, 2]
    
    if chunk_size is None:
        chunk_size = replay_buffer.chunk_size
    
    results = {
        'passed': True,
        'details': [],
        'mismatches': [],
    }
    
    logger.info("="*60)
    logger.info("Verifying Data Consistency: train_data vs replay_buffer")
    logger.info(f"  Chunk size: {chunk_size}")
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
    
    # Build episode ranges for action chunk construction
    episode_ids = replay_buffer.episode_ids[:replay_buffer.size]
    episode_ends = replay_buffer.episode_ends[:replay_buffer.size]
    end_indices = np.where(episode_ends)[0]
    
    # Build episode -> (start_idx, end_idx) mapping
    episode_ranges = {}
    prev_end = -1
    for end_idx in end_indices:
        start_idx = prev_end + 1
        ep_id = episode_ids[start_idx]
        episode_ranges[ep_id] = (start_idx, end_idx)
        prev_end = end_idx
    
    for idx in sample_indices:
        if idx >= len(dataset) or idx >= replay_buffer.size:
            logger.warning(f"Index {idx} out of range, skipping")
            continue
        
        logger.info(f"\n--- Comparing sample at index {idx} ---")
        
        # Get sample from dataset (already normalized/transformed by wrappers)
        ds_sample = dataset[idx]
        if data_processor is not None:
            ds_sample_processed = data_processor(ds_sample)
        else:
            ds_sample_processed = ds_sample
        
        # Build sample from replay buffer with action chunk (same logic as sample_processed)
        ep_id = int(episode_ids[idx])
        start_idx, end_idx = episode_ranges.get(ep_id, (idx, idx))
        ep_length = end_idx - start_idx + 1
        frame_offset = idx - start_idx
        
        # Build action chunk
        remaining_in_episode = ep_length - frame_offset
        valid_count = min(chunk_size, remaining_in_episode)
        
        actions = replay_buffer.actions
        if valid_count > 0:
            valid_end = idx + valid_count
            valid_actions = actions[idx:valid_end].copy()
            
            if valid_count < chunk_size:
                pad_count = chunk_size - valid_count
                last_action = valid_actions[-1:] if len(valid_actions) > 0 else actions[end_idx:end_idx+1]
                padding = np.repeat(last_action, pad_count, axis=0)
                action_chunk = np.concatenate([valid_actions, padding], axis=0)
                is_pad = np.array([False] * valid_count + [True] * pad_count, dtype=bool)
            else:
                action_chunk = valid_actions
                is_pad = np.array([False] * chunk_size, dtype=bool)
        else:
            last_action = actions[end_idx:end_idx+1]
            action_chunk = np.repeat(last_action, chunk_size, axis=0)
            is_pad = np.array([True] * chunk_size, dtype=bool)
        
        # Build rb_sample
        obs_storage = replay_buffer.obs_storage
        rb_sample = {}
        
        # Image
        if 'image' in obs_storage:
            rb_sample['image'] = torch.from_numpy(obs_storage['image'][idx])
        
        # State
        state_key = None
        for k in ['state', 'state_ee', 'state_joint']:
            if k in obs_storage:
                state_key = k
                break
        if state_key is not None:
            rb_sample['state'] = torch.from_numpy(obs_storage[state_key][idx].copy())
        
        # Action chunk
        rb_sample['action'] = torch.from_numpy(action_chunk.astype(np.float32))
        rb_sample['is_pad'] = torch.from_numpy(is_pad)
        
        # Metadata
        rb_sample['raw_lang'] = obs_storage.get('raw_lang', {}).get(idx, '') if isinstance(obs_storage.get('raw_lang'), dict) else (obs_storage['raw_lang'][idx] if 'raw_lang' in obs_storage else '')
        rb_sample['reasoning'] = ''
        rb_sample['episode_id'] = ep_id
        rb_sample['timestamp'] = int(replay_buffer.timestamps[idx]) if replay_buffer.timestamps is not None else frame_offset
        
        # Apply normalization if buffer stores raw data
        if replay_buffer.store_raw:
            rb_sample = apply_normalization_to_sample(
                rb_sample,
                action_normalizer=replay_buffer.action_normalizer,
                state_normalizer=replay_buffer.state_normalizer,
            )
            
            if replay_buffer.transforms is not None:
                transforms = replay_buffer.transforms
                if not isinstance(transforms, (list, tuple)):
                    transforms = [transforms]
                rb_sample = apply_transforms_to_sample(rb_sample, transforms)
        
        # Apply processor
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
    DataLoader-like wrapper for replay buffer with on-demand normalization and transforms.
    
    NEW SAMPLING STRATEGY:
    - Replay buffer stores single-step actions (no duplication)
    - Action chunks are built dynamically during sampling
    - Episode boundaries are respected for padding
    
    Workflow (when buffer stores raw data):
    1. Sample episodes and start positions
    2. Build action chunks with is_pad based on episode boundaries
    3. Apply normalization (state/action normalizers)
    4. Apply transforms (data augmentation, etc.)
    5. Apply data_processor (policy-specific processing)
    6. Collate using data_collator
    
    This provides maximum flexibility for:
    - Runtime data augmentation (can change transforms without reloading data)
    - Experimenting with different normalization strategies
    - Different chunk sizes at sampling time
    
    Memory Management:
    - Periodically runs gc.collect() to avoid memory buildup
    - Set gc_interval to control how often GC runs
    """
    
    def __init__(
        self,
        replay_buffer: ILReplayBuffer,
        batch_size: int,
        num_batches_per_epoch: int = 1000,
        data_processor=None,
        data_collator=None,
        device: str = "cuda:0",
        gc_interval: int = 5,
        apply_normalization: bool = True,
        apply_transforms: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            replay_buffer: ILReplayBuffer instance (stores single-step actions)
            batch_size: Number of samples per batch
            num_batches_per_epoch: Number of batches to generate per epoch
            data_processor: Policy-specific processor function
            data_collator: Collator function for batching
            device: Target device for batches
            gc_interval: Garbage collection frequency (0 to disable)
            apply_normalization: Whether to apply normalizers during sampling
            apply_transforms: Whether to apply transforms during sampling
            chunk_size: Action chunk size (default: replay_buffer.chunk_size)
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.data_processor = data_processor
        self.data_collator = data_collator
        self.device = device
        self.gc_interval = gc_interval
        self.apply_normalization = apply_normalization
        self.apply_transforms = apply_transforms
        self.chunk_size = chunk_size if chunk_size is not None else replay_buffer.chunk_size
        
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
        """Sample and process a batch with full pipeline.
        
        NEW: Samples episodes, builds action chunks dynamically with proper padding.
        Memory-optimized: Cleans up intermediate objects promptly.
        """
        # Get processed samples (episode sampling → action chunk building → normalization → transforms → processor)
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            self.data_processor,
            device='cpu',  # Keep on CPU, collator/move will handle device
            apply_normalization=self.apply_normalization,
            apply_transforms=self.apply_transforms,
            chunk_size=self.chunk_size,
        )
        
        # Collate
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        # Clear samples list immediately after collation (no longer needed)
        del samples
        
        # Move to device and return
        result = self._to_device(batch, self.device)
        
        # Clear batch dict to free memory (tensors are already moved to device)
        del batch
        
        return result
    
    def sample_batch_raw(self):
        """Sample a batch of raw data without any processing."""
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            data_processor=None,
            device='cpu',
            apply_normalization=False,
            apply_transforms=False,
        )
        
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        return self._to_device(batch, self.device)
    
    def sample_batch_normalized_only(self):
        """Sample a batch with only normalization (no transforms or processor)."""
        samples = sample_processed(
            self.replay_buffer, 
            self.batch_size, 
            data_processor=None,
            device='cpu',
            apply_normalization=True,
            apply_transforms=False,
        )
        
        if self.data_collator is not None:
            batch = self.data_collator(samples)
        else:
            batch = self._default_collate(samples)
        
        return self._to_device(batch, self.device)
    
    def set_transforms(self, transforms):
        """
        Update transforms at runtime.
        
        This allows changing data augmentation without reloading the buffer.
        
        Args:
            transforms: New transform pipeline (list of callables)
        """
        self.replay_buffer.transforms = transforms
    
    def set_normalizers(self, action_normalizer=None, state_normalizer=None):
        """
        Update normalizers at runtime.
        
        Args:
            action_normalizer: New action normalizer
            state_normalizer: New state normalizer
        """
        if action_normalizer is not None:
            self.replay_buffer.action_normalizer = action_normalizer
        if state_normalizer is not None:
            self.replay_buffer.state_normalizer = state_normalizer
    
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
    
    def get_data_info(self):
        """Get information about the data pipeline configuration."""
        return {
            'batch_size': self.batch_size,
            'num_batches_per_epoch': self.num_batches_per_epoch,
            'buffer_size': self.replay_buffer.size,
            'buffer_capacity': self.replay_buffer.capacity,
            'store_raw': self.replay_buffer.store_raw,
            'apply_normalization': self.apply_normalization,
            'apply_transforms': self.apply_transforms,
            'has_action_normalizer': self.replay_buffer.action_normalizer is not None,
            'has_state_normalizer': self.replay_buffer.state_normalizer is not None,
            'has_transforms': self.replay_buffer.transforms is not None,
            'has_processor': self.data_processor is not None,
            'has_collator': self.data_collator is not None,
        }


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
