"""
Base Replay Buffer Class

This module defines the base class for all replay buffers in the RL framework.

Design Philosophy:
- Store raw Meta data: Store original MetaObs and MetaAction in buffer, keeping data in its raw form
- Complete information storage: Support storing all fields of MetaObs and MetaAction
- Extensibility: Support storing additional custom fields (value, log_prob, advantage, trajectory_id, etc.)
- Conversion during sampling: Convert to ILStudio data pipeline format (normalization, etc.) during sampling
- Compatibility: Maintain data integrity while being compatible with ILStudio's normalization pipeline
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod


class BaseReplay(ABC):
    """
    Base class for Replay Buffer.
    
    This class defines the interface for all replay buffers in the RL framework.
    Replay buffers store experience data (transitions) for training RL algorithms.
    
    Attributes:
        capacity: Maximum number of transitions to store
        device: Device to store data on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ):
        """
        Initialize the Replay Buffer.
        
        Args:
            capacity: Buffer capacity (maximum number of transitions to store)
            device: Data storage device ('cpu' or 'cuda', default 'cpu')
                   - 'cpu': Store data in CPU memory
                   - 'cuda' or 'cuda:0': Store data in GPU memory
            **kwargs: Other initialization parameters
        """
        self.capacity = capacity
        self.device = torch.device(device) if isinstance(device, str) else device
        self._size = 0
        self._position = 0
    
    @abstractmethod
    def add(self, transition: Dict[str, Any]) -> None:
        """
        Add a transition to the buffer.
        
        Stores raw Meta data (MetaObs, MetaAction) without any normalization.
        Supports storing all fields of MetaObs and MetaAction, plus additional custom information.
        
        Args:
            transition: Dictionary containing the following fields:
                - state: MetaObs format current state (raw data, including all fields)
                - action: MetaAction format action (raw data, including all fields)
                - reward: float reward
                - next_state: MetaObs format next state (raw data)
                - done: bool whether episode ended
                - info: Optional, additional information dictionary
                - **other custom fields**: Can store any additional information
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a batch from the buffer (raw data).
        
        Args:
            batch_size: Batch size
        
        Returns:
            Dictionary containing raw Meta data (without normalization)
        """
        raise NotImplementedError
    
    def sample_for_training(
        self, 
        batch_size: int,
        data_processor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Sample and convert to ILStudio training format.
        
        Args:
            batch_size: Batch size
            data_processor: Optional data processing function to align with ILStudio pipeline
                          - If None, return raw data
                          - If provided, should be a function: batch -> processed_batch
        
        Returns:
            Processed batch data (conforming to ILStudio training format)
        """
        batch = self.sample(batch_size)
        if data_processor is not None:
            batch = data_processor(batch)
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the buffer."""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save buffer data to file.
        
        Args:
            path: Save path (can be file path or directory path)
            **kwargs: Save options
                - format: Save format (e.g., 'pkl', 'hdf5', 'npz', optional)
                - compress: Whether to compress (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str, **kwargs) -> None:
        """
        Load data from file into buffer.
        
        Args:
            path: Load path (can be file path or directory path)
            **kwargs: Load options
                - format: Load format (optional, can auto-detect)
                - append: Whether to append to existing buffer (default False, clear before load)
        """
        raise NotImplementedError
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size >= self.capacity
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all data in the buffer.
        
        Returns:
            Dictionary containing all stored data
        """
        return self.sample(self._size) if self._size > 0 else {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(capacity={self.capacity}, size={self._size}, device={self.device})"


if __name__ == '__main__':
    """
    Test code for BaseReplay class.
    
    Since BaseReplay is abstract, we create a simple concrete implementation for testing.
    """
    import sys
    sys.path.insert(0, '/home/zhang/robot/126/ILStudio')
    
    from benchmark.base import MetaObs, MetaAction
    from dataclasses import asdict
    
    # Simple concrete implementation for testing
    class SimpleReplay(BaseReplay):
        """Simple in-memory replay buffer for testing."""
        
        def __init__(self, capacity: int = 1000, device: str = 'cpu', **kwargs):
            super().__init__(capacity=capacity, device=device, **kwargs)
            self._storage = []
        
        def add(self, transition: Dict[str, Any]) -> None:
            if self._size < self.capacity:
                self._storage.append(transition)
                self._size += 1
            else:
                self._storage[self._position] = transition
            self._position = (self._position + 1) % self.capacity
        
        def sample(self, batch_size: int) -> Dict[str, Any]:
            if self._size == 0:
                return {}
            indices = np.random.randint(0, self._size, size=min(batch_size, self._size))
            batch = {
                'states': [self._storage[i]['state'] for i in indices],
                'actions': [self._storage[i]['action'] for i in indices],
                'rewards': np.array([self._storage[i]['reward'] for i in indices]),
                'next_states': [self._storage[i]['next_state'] for i in indices],
                'dones': np.array([self._storage[i]['done'] for i in indices]),
            }
            return batch
        
        def clear(self) -> None:
            self._storage = []
            self._size = 0
            self._position = 0
        
        def save(self, path: str, **kwargs) -> None:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self._storage[:self._size], f)
            print(f"Saved {self._size} transitions to {path}")
        
        def load(self, path: str, **kwargs) -> None:
            import pickle
            append = kwargs.get('append', False)
            if not append:
                self.clear()
            with open(path, 'rb') as f:
                data = pickle.load(f)
            for transition in data:
                self.add(transition)
            print(f"Loaded {len(data)} transitions from {path}")
    
    # Test the implementation
    print("=" * 60)
    print("Testing BaseReplay (SimpleReplay implementation)")
    print("=" * 60)
    
    # Create buffer
    buffer = SimpleReplay(capacity=100, device='cpu')
    print(f"\n1. Created buffer: {buffer}")
    
    # Create sample transitions
    print("\n2. Adding transitions...")
    for i in range(10):
        state = MetaObs(
            state=np.random.randn(10).astype(np.float32),
            state_ee=np.random.randn(7).astype(np.float32),
            raw_lang="pick up the red block"
        )
        action = MetaAction(
            action=np.random.randn(7).astype(np.float32),
            ctrl_space='ee',
            ctrl_type='delta'
        )
        next_state = MetaObs(
            state=np.random.randn(10).astype(np.float32),
            state_ee=np.random.randn(7).astype(np.float32),
            raw_lang="pick up the red block"
        )
        
        transition = {
            'state': asdict(state),
            'action': asdict(action),
            'reward': np.random.randn(),
            'next_state': asdict(next_state),
            'done': i == 9,
            'info': {'step': i},
            'value': np.random.randn(),  # Custom field
            'log_prob': np.random.randn(),  # Custom field
        }
        buffer.add(transition)
    
    print(f"   Buffer size after adding: {len(buffer)}")
    print(f"   Buffer is full: {buffer.is_full()}")
    
    # Sample from buffer
    print("\n3. Sampling from buffer...")
    batch = buffer.sample(batch_size=5)
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Batch rewards shape: {batch['rewards'].shape}")
    print(f"   Number of states in batch: {len(batch['states'])}")
    
    # Test sample_for_training with processor
    print("\n4. Testing sample_for_training with data processor...")
    def simple_processor(batch):
        """Simple processor that adds a 'processed' flag."""
        batch['processed'] = True
        return batch
    
    processed_batch = buffer.sample_for_training(batch_size=5, data_processor=simple_processor)
    print(f"   Processed batch has 'processed' key: {'processed' in processed_batch}")
    
    # Test save and load
    print("\n5. Testing save and load...")
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'buffer.pkl')
        buffer.save(save_path)
        
        # Create new buffer and load
        new_buffer = SimpleReplay(capacity=100)
        new_buffer.load(save_path)
        print(f"   Loaded buffer size: {len(new_buffer)}")
    
    # Test clear
    print("\n6. Testing clear...")
    buffer.clear()
    print(f"   Buffer size after clear: {len(buffer)}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

