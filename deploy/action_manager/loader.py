"""
Action Manager Loader

This module provides utilities to load action manager configurations from YAML files
or direct config objects and instantiate the corresponding manager class.

Supports dynamic class loading from config files without hardcoded registration.
"""

import yaml
import os
import importlib
from pathlib import Path
from typing import Dict, Any, Union
from types import SimpleNamespace


def load_action_manager(manager_name_or_path: str = None, config: Union[Dict, SimpleNamespace, Any] = None):
    """
    Load and instantiate an action manager.
    
    Args:
        manager_name_or_path: Can be:
            - A manager class name (e.g., 'BasicActionManager')
            - A config name (e.g., 'basic', 'older_first')
            - A path to a YAML config file
            - None (will use config parameter)
        config: Configuration object/dict containing manager settings.
                Can be command line args, a config dict, or SimpleNamespace.
    
    Returns:
        An instantiated action manager instance
    
    Examples:
        # Load from ConfigLoader (recommended)
        from configs.loader import ConfigLoader
        cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
        manager_cfg, _ = cfg_loader.load_manager('truncated_conservative')
        manager = load_action_manager(config=manager_cfg)
        
        # Load from config name (legacy, uses ConfigLoader internally)
        manager = load_action_manager('basic')
        
        # Load from config dict with dynamic class loading
        config = {
            'module_path': 'deploy.action_manager.truncated',
            'class_name': 'TruncatedManager',
            'start_ratio': 0.1,
            'end_ratio': 0.2
        }
        manager = load_action_manager(config=config)
    """
    # Import built-in managers for fallback
    from . import (
        BasicActionManager,
        OlderFirstManager,
        TemporalAggManager,
        TemporalOlderManager,
        DelayFreeManager,
        TruncatedManager
    )
    
    # Built-in manager name to class mapping (fallback only)
    BUILTIN_MANAGER_MAP = {
        'BasicActionManager': BasicActionManager,
        'basic': BasicActionManager,
        'OlderFirstManager': OlderFirstManager,
        'older_first': OlderFirstManager,
        'TemporalAggManager': TemporalAggManager,
        'temporal_agg': TemporalAggManager,
        'TemporalOlderManager': TemporalOlderManager,
        'temporal_older': TemporalOlderManager,
        'DelayFreeManager': DelayFreeManager,
        'delay_free': DelayFreeManager,
        'TruncatedManager': TruncatedManager,
        'truncated': TruncatedManager,
    }
    
    # Store original config dict if provided
    manager_config = {}
    manager_class = None
    manager_class_name = None
    
    # Priority 1: If config is a dict (from ConfigLoader), use it directly
    if isinstance(config, dict):
        manager_config = config.copy()
        
        # Try to dynamically load class from config
        module_path = manager_config.get('module_path')
        class_name = manager_config.get('class_name') or manager_config.get('manager_name') or manager_config.get('name')
        
        # Auto-detect module_path if class_name is a full path
        if class_name and '.' in class_name and not module_path:
            parts = class_name.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            
        if module_path and class_name:
            # Dynamic import from config
            try:
                module = importlib.import_module(module_path)
                manager_class = getattr(module, class_name)
                print(f"Dynamically loaded {class_name} from {module_path}")
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to load {class_name} from {module_path}: {e}")
        elif class_name:
            # Try builtin map
            manager_class = BUILTIN_MANAGER_MAP.get(class_name)
            if manager_class is None:
                raise ValueError(
                    f"Unknown action manager: {class_name}. "
                    f"Available built-in managers: {list(BUILTIN_MANAGER_MAP.keys())}\n"
                    f"Or specify 'module_path' in config for dynamic loading."
                )
    
    # Priority 2: Load from YAML file if manager_name_or_path is a file
    elif manager_name_or_path and (manager_name_or_path.endswith('.yaml') or manager_name_or_path.endswith('.yml')):
        # Use ConfigLoader to load with proper override support
        from configs.loader import ConfigLoader
        cfg_loader = ConfigLoader()
        manager_config, _ = cfg_loader.load_manager(manager_name_or_path)
        
        # Dynamic import
        module_path = manager_config.get('module_path')
        class_name = manager_config.get('class_name') or manager_config.get('manager_name') or manager_config.get('name')
        
        # Auto-detect module_path if class_name is a full path
        if class_name and '.' in class_name and not module_path:
            parts = class_name.rsplit('.', 1)
            module_path = parts[0]
            class_name = parts[1]
            
        if module_path and class_name:
            try:
                module = importlib.import_module(module_path)
                manager_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to load {class_name} from {module_path}: {e}")
        elif class_name:
            manager_class = BUILTIN_MANAGER_MAP.get(class_name)
            if manager_class is None:
                raise ValueError(f"Unknown action manager: {class_name}")
    
    # Priority 3: Manager name/class provided directly (legacy support)
    elif manager_name_or_path:
        manager_class_name = manager_name_or_path
        manager_class = BUILTIN_MANAGER_MAP.get(manager_class_name)
        
        if manager_class is None and '.' in manager_class_name:
            # Try to load as full path
            try:
                parts = manager_class_name.rsplit('.', 1)
                module_path = parts[0]
                class_name = parts[1]
                module = importlib.import_module(module_path)
                manager_class = getattr(module, class_name)
            except Exception:
                pass

        if manager_class is None:
            # Try eval as last resort fallback
            try:
                manager_class = eval(manager_class_name)
            except Exception as e:
                raise ValueError(
                    f"Unknown action manager: {manager_class_name}. "
                    f"Available managers: {list(BUILTIN_MANAGER_MAP.keys())}"
                ) from e
    
    # Priority 4: Extract from config object (SimpleNamespace/args - legacy)
    elif config:
        if isinstance(config, SimpleNamespace) or hasattr(config, '__dict__'):
            config_dict = vars(config)
            manager_class_name = config_dict.get('action_manager') or config_dict.get('manager_name') or config_dict.get('manager')
            
            if manager_class_name:
                manager_class = BUILTIN_MANAGER_MAP.get(manager_class_name)
                if manager_class is None:
                    raise ValueError(f"Unknown action manager: {manager_class_name}")
            else:
                # Default
                manager_class = BasicActionManager
                print("No action manager specified, using default: BasicActionManager")
            
            # Extract parameters
            for key in ['coef', 'manager_coef', 'older_coef', 'duration', 'start_ratio', 'end_ratio']:
                if key in config_dict and config_dict[key] is not None:
                    manager_config[key] = config_dict[key]
    else:
        # Default case
        manager_class = BasicActionManager
        print("No action manager specified, using default: BasicActionManager")
    
    # Convert config dict to namespace for manager initialization
    final_config = SimpleNamespace(**manager_config) if manager_config else SimpleNamespace()
    
    # Instantiate manager
    try:
        manager = manager_class(final_config)
        print(f"Successfully loaded action manager: {manager_class.__name__}")
        return manager
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {manager_class.__name__}: {e}"
        ) from e

