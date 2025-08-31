# eval_real.py (argparse version)
import yaml
import os
import traceback
import time
import transformers
import importlib
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
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from deploy.teleoperator.base import str2dtype, generate_shm_info
import h5py
import signal
import sys
import select  # <<< ADDED: For non-blocking input without threads
import termios # <<< ADDED: For terminal control
import tty     # <<< ADDED: For terminal control

# action_lock = threading.Lock()

@dataclass
class HyperArguments:
    # shm_name: str = 'teleop_action_buffer'
    shm_name: str=''
    action_dtype: str = 'float64'
    action_dim: int = 7
    robot_config: str = "configuration/robots/dummy.yaml"
    frequency: int = 100
    save_dir: str = 'data/debug'
    task: str = field(default="sim_transfer_cube_scripted")
    start_idx: int = 0
    num_rollout: int = 4
    max_timesteps: int = 400
    image_size: str = '(640, 480)'  # (width, height)
    ctrl_space: str = 'joint'
    ctrl_type: str = 'abs'
    camera_ids: str = '[0]'
    image_size_primary: str = "(640,480)"
    image_size_wrist: str = "(640,480)"
    camera_names: List[str] = field(
        default_factory=lambda: ['primary'],
        metadata={"help": "List of camera names", "nargs": "+"}
    )

# <<< ADDED: Non-blocking Keyboard Input Class (replaces thread) >>>
class KBHit:
    def __init__(self):
        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)
        # New terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        self.set_normal_term()

    def set_normal_term(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def set_curses_term(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def check(self):
        # Check if there is data available on stdin
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def getch(self):
        # Read a character from stdin
        return sys.stdin.read(1)

    def getarrow(self):
        # Read an arrow key sequence
        c1 = self.getch()
        if c1 == '\x1b': # ESC
            c2 = self.getch()
            c3 = self.getch()
            return c3
        return None
    
    def get_input(self):
        """Checks for Enter key press and consumes the input line."""
        if self.check():
            # Read all available characters to find a newline
            chars = ""
            while self.check():
                char = self.getch()
                if char == '\n' or char == '\r':
                    return chars.strip() # Return stripped line if Enter is pressed
                else:
                    chars += char
        return None # Return None if no Enter key was pressed

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def parse_param():
    parser = transformers.HfArgumentParser((HyperArguments,))
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                key, value = key.split('=', 1)
            else:
                value = True
            extra_args[key] = value
    model_args = {}
    for key, value in extra_args.items():
        try:
            value = _convert_to_type(value)
            if key.startswith('model.'):
                model_args[key[6:]] = value
            else:
                setattr(args, key, value)
        except ValueError as e:
            print(f"Warning: {e}")
    args.model_args = model_args
    return args

def make_robot(robot_cfg: Dict, args, max_connect_retry: int = 5):
    full_path = robot_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    RobotCls = getattr(module, class_name)
    print(f"Creating robot: {full_path}")
    robot_config = robot_cfg.get('config', {})
    robot = RobotCls(**robot_config, extra_args=args)
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot

class RobotController:
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
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
            print("Robot controller: Successfully connected to shared memory.")
        except FileNotFoundError:
            print(f"Robot controller: Error! Shared memory '{self.shm_name}' does not exist.")
            raise

    def run(self):
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
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'episode_{episode_id:04d}.hdf5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        obs_group = f.create_group('observations')
        if observations:
            for key in observations[0].keys():
                data_list = [obs[key] for obs in observations]
                try:
                    obs_group.create_dataset(key, data=np.stack(data_list))
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
    print(f"Episode {episode_id} data saved to {file_path}")

# Global event to signal all threads to shut down gracefully.
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    if not shutdown_event.is_set():
        print("\nCtrl+C detected! Shutting down gracefully...", flush=True)
        shutdown_event.set()

if __name__ == '__main__':
    set_seed(0)
    args = parse_param()
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize non-blocking keyboard input
    kb_hit = KBHit()
    kb_hit.set_curses_term()

    # --- 1. Create Real-World Environment ---
    print(f"Loading robot configuration from {args.robot_config}")
    with open(args.robot_config, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    robot = make_robot(robot_cfg, args)
    print("Robot successfully loaded.")
    
    robot_controller = None
    controller_process = None
    action_shm = None

    # --- 2. Connect to shm ---
    if args.shm_name:
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
    episode_count = args.start_idx
    try:
        rate_limiter = RateLimiter()
        while not shutdown_event.is_set():
            # Wait for user to start the episode
            print(f"\n{'='*10}\nPress Enter to START episode {episode_count}...\n{'='*10}")
            while not shutdown_event.is_set():
                if kb_hit.get_input() is not None:
                    break
                time.sleep(0.1) # Small sleep to prevent busy-waiting
            
            if shutdown_event.is_set(): break

            print(f"Starting episode {episode_count}. Recording...")
            
            observations, actions = [], []
            
            print("Press Enter to STOP recording...")
            # Consume any prior input
            while kb_hit.get_input() is not None: pass

            # Collection loop
            stop_recording = False
            all_timestamps = []
            while not stop_recording and not shutdown_event.is_set():
                if kb_hit.get_input() is not None:
                    stop_recording = True
                else:
                    obs = robot.get_observation()
                    current_time = time.perf_counter()
                    if obs:
                        obs['_timestamp'] = current_time
                        all_timestamps.append(current_time)
                        observations.append(obs)
                        if action_buffer is not None:
                            action = action_buffer[0]['action'].copy()
                            actions.append(action)
                        rate_limiter.sleep(args.frequency)

            if shutdown_event.is_set(): break
            actual_frequency = len(all_timestamps)/(all_timestamps[-1] - all_timestamps[0])
            print(f"Episode {episode_count} finished at {actual_frequency:.2f}Hz ({args.frequency}Hz expected). Collected {len(observations)} timesteps.")
            
            # Save the collected data
            print("Save this episode? (Press Enter to SAVE, or type anything and press Enter to DISCARD)")
            saving_prompt = None
            while saving_prompt is None and not shutdown_event.is_set():
                saving_prompt = kb_hit.get_input()
                if saving_prompt is None:
                    time.sleep(0.1)
            
            if shutdown_event.is_set(): break

            if len(saving_prompt) == 0:
                if observations:
                    save_episode_to_hdf5(args.save_dir, episode_count, observations, actions)
                    print(f"Episode {episode_count} was successfully saved to {args.save_dir}.")
                    episode_count += 1
                else:
                    print("No data collected, skipping save.")
            else:
                print("Discarding episode.")

    except KeyboardInterrupt:
        print("\n[Main Process] Exit by KeyboardInterrupt (fallback).")
    finally:
        # --- 5. Graceful Shutdown ---
        print("\n[Main Process] Shutting down...")
        shutdown_event.set()
        kb_hit.set_normal_term() # Restore terminal settings

        if robot_controller:
            robot_controller.stop()
            print("Stop signal sent to robot controller.")
            
        if controller_process and controller_process.is_alive():
            controller_process.join(timeout=2)
            if controller_process.is_alive():
                print("Controller process did not terminate gracefully, forcing termination.")
                controller_process.terminate()
            else:
                print("Robot controller process joined successfully.")

        if action_shm:
            action_shm.close()
            print("Main process shared memory link closed.")

        if robot:
            robot.shutdown()
            print("Robot shutdown command sent.")
            
        print("Cleanup complete. Exiting.")