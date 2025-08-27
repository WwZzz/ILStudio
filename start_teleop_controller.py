import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
import importlib
import argparse
import yaml
from deploy.teleoperator.base import str2dtype, BaseTeleopDevice

def load_teleoperator(teleop_cfg: dict, args, shm_info: dict) -> BaseTeleopDevice:
    """
    Dynamically load a teleoperator class specified in the configuration file.

    Args:
        teleop_cfg : dict
            Teleoperator configuration dictionary. Must contain a 'target' key
            pointing to the fully-qualified class name, e.g. 'some.module.TeleOpClass'.
        args : argparse.Namespace
            Command-line arguments.
        shm_info : dict
            Shared-memory metadata containing 'name', 'shape', 'dtype', and 'size'.

    Returns:
        teleoperator : BaseTeleopDevice
            An instantiated teleoperator object ready to run.
    """
    full_path = teleop_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    TeleOpCls = getattr(module, class_name)
    print(f"Creating Teleop Device: {full_path}")
    teleoperator = TeleOpCls(
        shm_name=shm_info['name'],
        shm_shape=shm_info['shape'],
        shm_dtype=shm_info['dtype'],
        action_dim=args.action_dim,
        action_dtype=args.action_dtype,
        frequency=args.freq,
        **{k: v for k, v in teleop_cfg.items() if k != 'target'}
    )
    return teleoperator


def main():
    parser = argparse.ArgumentParser(description='Teleoperation parameters')

    parser.add_argument(
        '--tele-config', type=str, default='configuration/teleop/keyboard.yaml',
        help='YAML file describing the teleoperator configuration'
    )
    parser.add_argument(
        '--shm-name', type=str, default='teleop_action_buffer',
        help='Name of the shared-memory buffer used for actions'
    )
    parser.add_argument(
        '--action-dim', type=int, default=7,
        help='Dimensionality of the action space'
    )
    parser.add_argument(
        '--action-dtype', type=str, default='float64', choices=['float32', 'float64'],
        help='Numpy dtype used for action values'
    )
    parser.add_argument(
        '--freq', type=float, default=10.0,
        help='Teleoperation update frequency in Hz'
    )
    args = parser.parse_args()

    # Convert the string dtype to an actual numpy dtype
    args.action_dtype = str2dtype(args.action_dtype)

    # Define the shared-memory layout
    shm_info = {
        'name': args.shm_name,
        'dtype': np.dtype([
            ('timestamp', np.float64),
            ('action', args.action_dtype, args.action_dim),
        ]),
        'shape': (1,),
    }
    shm_info['size'] = shm_info['dtype'].itemsize

    # Create or attach to the shared-memory block
    try:
        shm = shared_memory.SharedMemory(
            name=shm_info['name'],
            create=True,
            size=shm_info['size']
        )
        print(f"Main: created shared-memory '{shm_info['name']}' "
              f"({shm_info['size']} bytes)")
    except FileExistsError:
        print(f"Main: shared-memory '{shm_info['name']}' already exists; attaching")
        shm = shared_memory.SharedMemory(name=shm_info['name'])

    # Load teleoperator configuration and instantiate the device
    print(f"Loading teleop configuration from {args.tele_config}")
    with open(args.tele_config, 'r') as f:
        teleop_cfg = yaml.safe_load(f)

    teleop_dev = load_teleoperator(teleop_cfg, args, shm_info)
    if hasattr(teleop_dev, 'get_doc'):
        print(teleop_dev.get_doc())

    try:
        # Start the teleoperation loop
        teleop_dev.run()
    except KeyboardInterrupt:
        print("\nMain: Ctrl+C received; requesting all processes to stop...")
    finally:
        teleop_dev.stop()
        print("Main: all processes have stopped.")
        shm.close()
        shm.unlink()
        print(f"Main: shared-memory '{shm_info['name']}' has been cleaned up.")


if __name__ == '__main__':
    main()