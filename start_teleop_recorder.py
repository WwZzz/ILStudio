# eval_real.py (argparse version)
import yaml
import os
import traceback
import time
import transformers
import importlib
import threading
import queue
import torch
import numpy as np
from benchmark.base import MetaPolicy
from data_utils.utils import set_seed, _convert_to_type, load_normalizers
from deploy.robot.base import AbstractRobotInterface, RateLimiter
from PIL import Image, ImageDraw, ImageFont
from configuration.utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
from configuration.constants import TASK_CONFIGS
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from deploy.teleoperator.base import str2dtype, generate_shm_info
import h5py

# action_lock = threading.Lock()

@dataclass
class HyperArguments:
    shm_name: str = 'teleop_action_buffer'
    action_dtype: str = 'float64'
    action_dim: int = 7
    robot_config: str = "configuration/robots/dummy.yaml"
    frequency: int = 30
    # publish_rate determines how frequently actions are consumed.  
    # If too slow, actions may be overwritten before being used (especially when the buffer is small), causing large jitters.  
    # If too fast, the action producer cannot keep up, causing many stalls.  
    # Additionally, if the robot cannot execute as fast as the consumption rate, actions will be sent but not fully executed, wasting them.
    save_dir: str = 'data/debug'
    task: str = field(default="sim_transfer_cube_scripted")

    num_rollout: int = 4
    max_timesteps: int = 400
    image_size: str = '(640, 480)'  # (width, height)
    ctrl_space: str = 'joint'
    ctrl_type: str = 'abs'
    camera_ids: str = '[0]'
    # ############ data ###################
    image_size_primary: str = "(640,480)"  # image size of non-wrist camera
    image_size_wrist: str = "(640,480)"  # image size of wrist camera
    camera_names: List[str] = field(
        default_factory=lambda: ['primary'],
        metadata={"help": "List of camera names", "nargs": "+"}
    )

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def parse_param():
    global local_rank
    # Use HFParser to pass parameters, which are defined in the dataclass above
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    print(unknown_args)
    print(args)
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]  # Remove '--' prefix
            if "=" in key:
                key, value = key.split('=', 1)
            else:
                value = True  # If no value is specified (e.g., --flag), default to True
            extra_args[key] = value
    model_args = {}
    # Dynamically inject `extra_args` into the args object
    for key, value in extra_args.items():
        try:
            value = _convert_to_type(value)
            if key.startswith('model.'):
                model_args[key[6:]] = value  # Dynamically get custom model_args, i.e., strings starting with model.
            else:
                setattr(args, key, value)  # Set non-model-related parameters as attributes of args
        except ValueError as e:
            print(f"Warning: {e}")
    args.model_args = model_args
    return args


def make_robot(robot_cfg: Dict, args, max_connect_retry: int = 5):
    """
    Factory function to create a robot instance from a config dictionary.

    Args:
        robot_cfg (Dict): A dictionary loaded from the robot's YAML config file.
    """
    full_path = robot_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    RobotCls = getattr(module, class_name)
    print(f"Creating robot: {full_path}")

    # .get() is used to safely access params, which might not exist
    robot_config = robot_cfg.get('config', {})

    robot = RobotCls(**robot_config, extra_args=args)
    # connect to robot
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot


class RobotController:
    """
    Reads actions from shared memory and sends them to the robot.
    """

    def __init__(self, robot, shm_name, shm_shape, shm_dtype):
        self.robot = robot
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.shm = None
        self.action_buffer = None
        self.last_timestamp = 0.0
        self.stop_event = mp.Event()
        if not self.is_connected():
            self.connect_to_buffer()
    
    def is_connected(self):
        return self.shm is not None

    def connect_to_buffer(self):
        """Connect to shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("Robot controller: Successfully connected to shared memory.")
        except FileNotFoundError:
            print(f"Robot controller: Error! Shared memory '{self.shm_name}' does not exist.")
            raise

    def run(self):
        """
        Main loop: non-blocking read from buffer and send to the robot.
        """
        try:
            while not self.stop_event.is_set():
                current_timestamp = self.action_buffer[0]['timestamp']
                if current_timestamp > self.last_timestamp:
                    self.last_timestamp = current_timestamp
                    action = self.action_buffer[0]['action'].copy()
                    self.robot.publish_action(action)
        finally:
            print("\nRobot controller: Shutting down...")
            if self.shm:
                self.shm.close()

    def stop(self):
        self.stop_event.set()

def save_episode_to_hdf5(save_dir, episode_id, observations, actions):
    """Saves a single episode's data to an HDF5 file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'episode_{episode_id}.hdf5')
    
    with h5py.File(file_path, 'w') as f:
        # Save actions
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        
        # Save observations (assuming obs is a dictionary)
        # This handles nested dictionaries of numpy arrays, which is common.
        obs_group = f.create_group('observations')
        if observations:
            # Get keys from the first observation dictionary
            for key in observations[0].keys():
                # Stack all values for this key across all timesteps
                data_list = [obs[key] for obs in observations]
                try:
                    obs_group.create_dataset(key, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")

    print(f"Episode {episode_id} data saved to {file_path}")

if __name__ == '__main__':
    set_seed(0)
    args = parse_param()

    # --- 1. Create Real-World Environment ---
    # Load the robot-specific configuration from the provided YAML file
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    robot = make_robot(robot_cfg, args)
    print("Robot successfully loaded.")

    # --- 2. Connect to shm
    if args.shm_name !='':
        args.action_dtype = str2dtype(args.action_dtype)
        shm_info = generate_shm_info(args.shm_name, args.action_dim, args.action_dtype)
        robot_controller = RobotController(
            robot=robot,
            shm_name=shm_info['name'],
            shm_shape=shm_info['shape'],
            shm_dtype=shm_info['dtype'],
        )
        controller_process = mp.Process(target=robot_controller.run)
        controller_process.start()
        print("Robot controller process started in the background.")
        time.sleep(1)
        # The main process also needs to connect to shared memory to LOG actions.
        action_shm = None
        action_buffer = None
        try:
            action_shm = shared_memory.SharedMemory(name=shm_info['name'])
            action_buffer = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=action_shm.buf)
            print("Main process connected to shared memory for logging.")
        except (FileNotFoundError, TypeError):
            print("Warning: Could not connect to shared memory for logging. Actions will not be saved.")
    else:
        action_buffer = None
    
    # --- 4. Main Data Collection Loop ---
    episode_count = 0
    try:
        rate_limiter = RateLimiter()
        while True:
            # Wait for user to start the episode
            input(f"\n{'='*10}\nPress Enter to START episode {episode_count}...\n{'='*10}")
            print(f"Starting episode {episode_count}. Recording...")
            
            # Data storage for the current episode
            observations = []
            actions = []
            
            # Non-blocking input to stop collection
            stop_event = threading.Event()
            def wait_for_enter():
                input("Press Enter to STOP recording...")
                stop_event.set()

            input_thread = threading.Thread(target=wait_for_enter)
            input_thread.start()
            
            # Collection loop
            while not stop_event.is_set():
                # Get the latest observation from the robot
                # NOTE: We assume robot.get_obs() returns a dictionary of numpy arrays
                obs = robot.get_observation() 
                observations.append(obs)
                
                # Get the latest action from shared memory
                if action_buffer is not None:
                    action = action_buffer[0]['action'].copy()
                    actions.append(action)
                
                # Control the loop frequency (e.g., 50 Hz)
                rate_limiter.sleep(args.frequency)

            print(f"Episode {episode_count} finished. Collected {len(observations)} timesteps.")
            
            # Save the collected data
            saving_prompt = input("Saving this episode? (Enter/ n+Enter)").lower()
            if len(saving_prompt.strip())>0:
                if observations:
                    save_episode_to_hdf5(args.save_dir, episode_count, observations, actions)
                    episode_count += 1
                    
                else:
                    print("No data collected, skipping save.")

    except KeyboardInterrupt:
        print("\n[Main Process] Exit by KeyboardInterrupt (Ctrl+C).")
    finally:
        # --- 5. Graceful Shutdown ---
        print("\n[Main Process] Shutting down...")
        
        # Signal the controller process to stop
        if robot_controller:
            robot_controller.stop()
            print("Stop signal sent to robot controller.")
            
        # Wait for the controller process to terminate
        if 'controller_process' in locals() and controller_process.is_alive():
            controller_process.join(timeout=2)
            if controller_process.is_alive():
                print("Controller process did not terminate gracefully, forcing termination.")
                controller_process.terminate()
            else:
                print("Robot controller process joined successfully.")

        # Close shared memory connection in the main process
        if action_shm:
            action_shm.close()
            print("Main process shared memory link closed.")

        # Shut down the robot connection
        if robot:
            robot.shutdown()
            print("Robot shutdown command sent.")
            
        print("Cleanup complete. Exiting.")
