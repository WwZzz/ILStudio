import configs 
import os
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
import importlib
import argparse
import yaml
from loguru import logger
from configs.utils import resolve_yaml, parse_overrides, apply_overrides_to_mapping
from data_utils.utils import _convert_to_type
from deploy.teleoperator.base import str2dtype, BaseTeleopDevice, generate_shm_info, dtype2code

def load_teleoperator(teleop_cfg: dict, shm_info: dict, action_dim: int, action_dtype, freq: float) -> BaseTeleopDevice:
    """
    Dynamically load a teleoperator class specified in the configuration file.

    Args:
        teleop_cfg : dict
            Teleoperator configuration dictionary. Must contain a 'target' key
            pointing to the fully-qualified class name, e.g. 'some.module.TeleOpClass'.
        shm_info : dict
            Shared-memory metadata containing 'name', 'shape', 'dtype', and 'size'.
        action_dim : int
            Dimensionality of the action space.
        action_dtype : numpy.dtype
            Numpy dtype used for action values.
        freq : float
            Teleoperation update frequency in Hz.

    Returns:
        teleoperator : BaseTeleopDevice
            An instantiated teleoperator object ready to run.
    """
    full_path = teleop_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    TeleOpCls = getattr(module, class_name)
    logger.info(f"Creating Teleop Device: {full_path}")
    # Filter out 'target' and any keys that are already explicitly passed
    excluded_keys = {'target', 'shm_name', 'action_dim', 'action_dtype', 'freq'}
    teleop_kwargs = {k: v for k, v in teleop_cfg.items() if k not in excluded_keys}
    
    teleoperator = TeleOpCls(
        shm_name=shm_info['name'],
        shm_shape=shm_info['shape'],
        shm_dtype=shm_info['dtype'],
        action_dim=action_dim,
        action_dtype=action_dtype,
        frequency=freq,
        **teleop_kwargs
    )
    return teleoperator


def main():
    parser = argparse.ArgumentParser(description='Teleoperation parameters')

    parser.add_argument(
        "-c", '--config', type=str, default='keyboard',
        help='Teleop config (name under configs/teleop or absolute path to yaml)'
    )
    
    # Parse only the teleop config file argument
    args, unknown = parser.parse_known_args()
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown)

    # Load teleoperator device configuration
    try:
        cfg_path = cfg_loader._resolve('teleop', args.config)
    except Exception:
        cfg_path = args.config
    logger.info(f"Loading teleop device configuration from {cfg_path}")
    with open(cfg_path, 'r') as f:
        teleop_cfg = yaml.safe_load(f)

    # apply dotted overrides --teleop.xxx
    apply_overrides_to_mapping(teleop_cfg, cfg_loader.get_overrides('teleop'), _convert_to_type)
    
    # Extract parameters directly from teleop config
    shm_name = teleop_cfg.get('shm_name', 'ilstd_teleop_controller')
    action_dim = teleop_cfg.get('action_dim', 7)
    action_dtype = str2dtype(teleop_cfg.get('action_dtype', 'float64'))
    freq = teleop_cfg.get('freq', 10.0)

    # Define the shared-memory layout
    shm_info = generate_shm_info(shm_name, action_dim, action_dtype)

    # Create or attach to the shared-memory block
    try:
        shm = shared_memory.SharedMemory(
            name=shm_info['name'],
            create=True,
            size=shm_info['size']
        )
        # Initialize metadata fields
        shm_array = np.ndarray((1,), dtype=shm_info['dtype'], buffer=shm.buf)
        shm_array['action_dim'][0] = action_dim
        shm_array['action_dtype_code'][0] = dtype2code(action_dtype)
        logger.info(f"Main: created shared-memory '{shm_info['name']}' "
              f"({shm_info['size']} bytes) with action_dim={action_dim}, action_dtype={action_dtype}")
    except FileExistsError:
        logger.info(f"Main: shared-memory '{shm_info['name']}' already exists; attaching")
        shm = shared_memory.SharedMemory(name=shm_info['name'])

    # Instantiate the teleoperator device using teleop configuration
    teleop_dev = load_teleoperator(teleop_cfg, shm_info, action_dim, action_dtype, freq)
    if hasattr(teleop_dev, 'get_doc'):
        logger.info(teleop_dev.get_doc())

    try:
        # Start the teleoperation loop
        teleop_dev.run()
    except KeyboardInterrupt:
        logger.info("Main: Ctrl+C received; requesting all processes to stop...")
    finally:
        teleop_dev.stop()
        logger.info("Main: all processes have stopped.")
        shm.close()
        shm.unlink()
        logger.info(f"Main: shared-memory '{shm_info['name']}' has been cleaned up.")


if __name__ == '__main__':
    main()