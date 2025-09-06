"""
Task configuration loader utilities.
"""

import yaml
import os


def load_task_config(task_name, config_dir=None):
    """
    Load a task configuration from a YAML file.
    
    Args:
        task_name (str): Name of the task (corresponds to YAML filename)
        config_dir (str): Directory containing YAML files (default: 'configs/task')
    
    Returns:
        dict: Task configuration
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__))
    yaml_path = os.path.join(config_dir, f'{task_name}.yaml')
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
