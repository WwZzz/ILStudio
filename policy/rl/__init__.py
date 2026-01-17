"""
Reinforcement Learning module for ILStudio.

This module provides RL training support for both offline and online algorithms.

Key Components:
===============

1. **ILReplayBuffer**: Replay buffer using MetaObs/MetaAction format:
   - Stores observations in MetaObs format (consistent with ILStudio)
   - Stores actions in MetaAction format with chunk support
   - Integrates action_normalizer and state_normalizer for standardization
   - Supports importing from offline datasets
   - Supports adding online interaction data

2. **RLPolicy**: Base class for RL policies with:
   - Standard forward/select_action/update interface
   - Compatible with ILStudio's policy loader
   - Save/load checkpoint support

3. **RLTrainer**: Base trainer class supporting:
   - OFFLINE mode: Train from offline dataset (BC, IQL, CQL, TD3+BC)
   - ONLINE mode: Train via environment interaction (SAC, TD3, PPO)
   - HYBRID mode: Offline pretraining + online finetuning

4. **TransitionConverter**: Utilities for converting between:
   - ILStudio dataset samples ↔ RL transitions
   - Gym observations ↔ Policy states
   - Policy outputs ↔ Environment actions

Usage Example:
==============

Offline RL Training (compatible with train.py):
```python
from policy.rl import ILReplayBuffer, RLConfig, OfflineRLTrainer
from data_utils.utils import load_data

# Load offline data (same as IL training)
data_dict = load_data(args, task_config)
train_data = data_dict['train']

# Create replay buffer from dataset with normalizers
replay_buffer = ILReplayBuffer.from_ilstudio_dataset(
    dataset=train_data,
    capacity=1000000,
    chunk_size=1,  # or larger for action chunks
    action_normalizer=action_normalizer,
    state_normalizer=state_normalizer,
    ctrl_space='ee',
    ctrl_type='delta',
)

# Create trainer and train
config = RLConfig(mode=RLMode.OFFLINE, total_steps=100000)
trainer = OfflineRLTrainer(config, policy, replay_buffer=replay_buffer)
trainer.train()
```

Online RL Training:
```python
from policy.rl import ILReplayBuffer, RLConfig, OnlineRLTrainer

# Create environment
env = create_env(env_config)

# Create buffer with normalizers
buffer = ILReplayBuffer(
    capacity=1000000,
    chunk_size=1,
    action_normalizer=action_normalizer,
    state_normalizer=state_normalizer,
)

# Collect data by interacting with environment
obs = env.reset()
for t in range(1000):
    action = policy.select_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    buffer.add_from_env_step(obs, action, reward, next_obs, done, truncated, info)
    obs = next_obs if not done else env.reset()

# Sample for training
batch = buffer.sample(batch_size=256, normalize=True)
```

Test the buffer with environments:
```bash
python -m policy.rl.replay_buffer
```

Directory Structure:
====================
policy/rl/
├── __init__.py          # This file - exports and documentation
├── replay_buffer.py     # ILReplayBuffer implementation (MetaObs/MetaAction)
├── base.py              # RLPolicy, RLTrainer, RLConfig base classes
├── transitions.py       # Transition conversion utilities
├── algorithms/          # RL algorithm implementations (future)
│   ├── offline/         # Offline RL (BC, IQL, CQL, TD3+BC)
│   ├── online/          # Online RL (SAC, TD3, PPO)
│   └── hybrid/          # Hybrid algorithms
└── configs/             # Algorithm-specific configs (future)
"""

# Core components
from .replay_buffer import (
    ILReplayBuffer,
    BatchTransition,
    RLTransition,
)

from .base import (
    RLPolicy,
    RLTrainer,
    RLConfig,
    RLMode,
    OfflineRLTrainer,
    OnlineRLTrainer,
    HybridRLTrainer,
)

from .transitions import (
    TransitionConverter,
    create_dataset_transition_iterator,
    batch_transitions,
)

__all__ = [
    # Replay Buffer
    'ILReplayBuffer',
    'BatchTransition',
    'RLTransition',
    
    # Base Classes
    'RLPolicy',
    'RLTrainer',
    'RLConfig',
    'RLMode',
    'OfflineRLTrainer',
    'OnlineRLTrainer',
    'HybridRLTrainer',
    
    # Transitions
    'TransitionConverter',
    'create_dataset_transition_iterator',
    'batch_transitions',
]
