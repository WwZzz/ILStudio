
import yaml
import os

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'  # Reset to default color

def load_task_config(task_name, config_dir=None):
	"""
	Load a task configuration from a YAML file.
	Args:
		task_name (str): Name of the task (corresponds to YAML filename)
		config_dir (str): Directory containing YAML files (default: 'configuration/task')
	Returns:
		dict: Task configuration
	"""
	if config_dir is None:
		config_dir = os.path.join(os.path.dirname(__file__), 'task')
	yaml_path = os.path.join(config_dir, f'{task_name}.yaml')
	with open(yaml_path, 'r') as f:
		return yaml.safe_load(f)