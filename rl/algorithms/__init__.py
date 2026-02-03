"""
RL Algorithms Module

This module provides various RL algorithm implementations.

Available algorithms:
- PPO: Proximal Policy Optimization
- SAC: Soft Actor-Critic
- TD3: Twin Delayed DDPG
- DPO: Direct Preference Optimization (for VLA)
- GRPO: Group Relative Policy Optimization (for VLA)
- REINFORCE: Basic policy gradient

Note: Implementations are provided in separate files.
This __init__.py provides factory functions for creating algorithms.
"""

from typing import Type, Dict, Any

# Registry for algorithm classes
_ALGORITHM_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str, algorithm_class: Type) -> None:
    """
    Register an algorithm class.
    
    Args:
        name: Algorithm name (e.g., 'ppo', 'sac')
        algorithm_class: Algorithm class to register
    """
    _ALGORITHM_REGISTRY[name.lower()] = algorithm_class


def get_algorithm_class(name_or_type: str) -> Type:
    """
    Get algorithm class by name or type string.
    
    Args:
        name_or_type: Algorithm name (e.g., 'ppo') or full type path 
                     (e.g., 'rl.algorithms.ppo.PPOAlgorithm')
    
    Returns:
        Algorithm class
    
    Raises:
        ValueError: If algorithm not found
    """
    # First check registry
    if name_or_type.lower() in _ALGORITHM_REGISTRY:
        return _ALGORITHM_REGISTRY[name_or_type.lower()]
    
    # Try to import from type path
    if '.' in name_or_type:
        try:
            parts = name_or_type.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import algorithm from '{name_or_type}': {e}")
    
    raise ValueError(f"Unknown algorithm: '{name_or_type}'. Available: {list(_ALGORITHM_REGISTRY.keys())}")


def list_algorithms() -> list:
    """List all registered algorithms."""
    return list(_ALGORITHM_REGISTRY.keys())


__all__ = [
    'register_algorithm',
    'get_algorithm_class',
    'list_algorithms',
]


# Auto-register built-in algorithms when this module is imported
def _register_builtin_algorithms():
    """Import algorithm submodules to trigger their registration."""
    try:
        from . import td3  # noqa: F401
    except ImportError:
        pass
    # Add more algorithms here as they are implemented
    # try:
    #     from . import sac
    # except ImportError:
    #     pass


_register_builtin_algorithms()
