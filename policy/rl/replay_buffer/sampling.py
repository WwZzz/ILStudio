"""
Sampling utilities for replay buffer.

This module provides functions for sampling from replay buffer with
action chunk building, normalization, and transforms.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from loguru import logger

from .utils import apply_normalization_to_sample, apply_transforms_to_sample


def build_action_chunk_from_buffer(
    actions: np.ndarray,
    global_idx: int,
    chunk_size: int,
    start_idx: int,
    end_idx: int,
    frame_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an action chunk from replay buffer with proper padding at episode boundaries.
    
    This function handles the logic of:
    1. Extracting valid actions from the buffer
    2. Padding with the last valid action if needed (when approaching episode end)
    3. Generating is_pad mask to indicate which actions are padded
    
    Args:
        actions: Action array from replay buffer (shape: [buffer_size, action_dim])
        global_idx: Global index in the buffer where to start building the chunk
        chunk_size: Desired chunk size
        start_idx: Start index of the episode containing global_idx
        end_idx: End index of the episode containing global_idx
        frame_offset: Offset of global_idx within the episode (global_idx - start_idx)
        
    Returns:
        Tuple of (action_chunk, is_pad):
        - action_chunk: np.ndarray of shape (chunk_size, action_dim) - the action chunk
        - is_pad: np.ndarray of shape (chunk_size,) - boolean mask indicating padded actions
    """
    ep_length = end_idx - start_idx + 1
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
        # All padding (shouldn't happen normally, but handle gracefully)
        last_action = actions[end_idx:end_idx+1]
        action_chunk = np.repeat(last_action, chunk_size, axis=0)
        is_pad = np.array([True] * chunk_size, dtype=bool)
    
    return action_chunk, is_pad


def build_sample_from_buffer(
    replay_buffer,
    global_idx: int,
    action_chunk: np.ndarray,
    is_pad: np.ndarray,
    episode_id: int,
    frame_offset: int,
    timestamps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build a sample dictionary from replay buffer data.
    
    This function extracts observation data (image, state), action chunk, and metadata
    from the replay buffer at a given index and constructs a sample dictionary in the
    format expected by ILStudio's data pipeline.
    
    Args:
        replay_buffer: The replay buffer instance
        global_idx: Global index in the buffer
        action_chunk: Action chunk array (shape: [chunk_size, action_dim])
        is_pad: Padding mask array (shape: [chunk_size])
        episode_id: Episode ID for this sample
        frame_offset: Offset within the episode (for timestamp fallback)
        timestamps: Optional timestamps array (default: replay_buffer.timestamps)
        
    Returns:
        Sample dictionary with keys: image, state, action, is_pad, raw_lang, reasoning, episode_id, timestamp
    """
    obs_storage = replay_buffer.obs_storage
    
    # Build sample dict
    sample = {}
    
    # Image
    if 'image' in obs_storage:
        sample['image'] = torch.from_numpy(obs_storage['image'][global_idx])
    
    # State - check for state, state_ee, or state_joint
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
        raw_lang_val = obs_storage['raw_lang'][global_idx]
        sample['raw_lang'] = raw_lang_val if raw_lang_val else ''
    else:
        sample['raw_lang'] = ''
    
    sample['reasoning'] = ''
    sample['episode_id'] = int(episode_id)
    
    # Timestamp - use provided timestamps or fallback to frame_offset
    if timestamps is not None:
        sample['timestamp'] = int(timestamps[global_idx])
    elif hasattr(replay_buffer, 'timestamps') and replay_buffer.timestamps is not None:
        sample['timestamp'] = int(replay_buffer.timestamps[global_idx])
    else:
        sample['timestamp'] = frame_offset
    
    return sample


def sample_processed(
    replay_buffer, 
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
    episode_ids = replay_buffer.episode_ids[:replay_buffer.size]
    episode_ends = replay_buffer.episode_ends[:replay_buffer.size]
    
    # Pre-compute episode boundaries (end indices where episode_ends=True)
    end_indices = np.where(episode_ends)[0]
    
    # Build episode -> (start_idx, end_idx) mapping
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
    
    samples = []
    
    for _ in range(batch_size):
        # Step 1: Randomly sample an episode
        ep_id = np.random.choice(list(episode_ranges.keys()))
        start_idx, end_idx = episode_ranges[ep_id]
        ep_length = end_idx - start_idx + 1
        
        # Step 2: Randomly sample start position within episode
        frame_offset = np.random.randint(0, ep_length)
        global_idx = start_idx + frame_offset
        
        # Step 3: Build action chunk with padding (using shared function)
        action_chunk, is_pad = build_action_chunk_from_buffer(
            actions=actions,
            global_idx=global_idx,
            chunk_size=chunk_size,
            start_idx=start_idx,
            end_idx=end_idx,
            frame_offset=frame_offset,
        )
        
        # Step 4: Build sample dict (using shared function)
        sample = build_sample_from_buffer(
            replay_buffer=replay_buffer,
            global_idx=global_idx,
            action_chunk=action_chunk,
            is_pad=is_pad,
            episode_id=ep_id,
            frame_offset=frame_offset,
            timestamps=timestamps,
        )
        
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
    replay_buffer,
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
    from .utils import apply_normalization_to_sample, apply_transforms_to_sample
    
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
        
        # Build sample from replay buffer with action chunk (using shared function)
        ep_id = int(episode_ids[idx])
        start_idx, end_idx = episode_ranges.get(ep_id, (idx, idx))
        frame_offset = idx - start_idx
        
        # Build action chunk using shared function
        actions = replay_buffer.actions
        action_chunk, is_pad = build_action_chunk_from_buffer(
            actions=actions,
            global_idx=idx,
            chunk_size=chunk_size,
            start_idx=start_idx,
            end_idx=end_idx,
            frame_offset=frame_offset,
        )
        
        # Build rb_sample using shared function
        rb_sample = build_sample_from_buffer(
            replay_buffer=replay_buffer,
            global_idx=idx,
            action_chunk=action_chunk,
            is_pad=is_pad,
            episode_id=ep_id,
            frame_offset=frame_offset,
            timestamps=replay_buffer.timestamps,
        )
        
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

