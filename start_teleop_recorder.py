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
from configs.task.loader import load_task_config
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, Sequence, List, Any
import time
import threading
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from deploy.teleoperator.base import str2dtype, generate_shm_info
import h5py
import signal
import sys
import os
import sys
if os.name == 'nt':
    import msvcrt  # Windows only
else:
    import select  # Unix only
    import termios # Unix only

# action_lock = threading.Lock()

@dataclass
class HyperArguments:
    # shm_name: str = 'teleop_action_buffer'
    shm_name: str=''
    action_dtype: str = 'float64'
    action_dim: int = 7
    config: str = "configs/robots/dummy.yaml"
    frequency: int = 100
    save_dir: str = 'data\\pick_red_cap_into_cup'
    task: str = field(default="sim_transfer_cube_scripted")
    start_idx: int = 0
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

class KBHit:
    def __init__(self):
        self.chars = ""
        if os.name == 'nt':
            # Windows: no terminal settings needed
            pass
        else:
            # Unix: Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            self.set_normal_term()

    def set_normal_term(self):
        if os.name != 'nt':
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def set_curses_term(self):
        if os.name != 'nt':
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def check(self):
        if os.name == 'nt':
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def getch(self):
        if os.name == 'nt':
            return msvcrt.getwch()
        else:
            return sys.stdin.read(1)

    def getarrow(self):
        c1 = self.getch()
        if c1 == '\x1b': # ESC
            c2 = self.getch()
            c3 = self.getch()
            return c3
        return None

    def get_input(self):
        if os.name == 'nt':
            while self.check():
                char = self.getch()
                if char in ('\r', '\n'):
                    res = self.chars.strip()
                    self.chars = ""
                    return res
                else:
                    self.chars += char
                    print("Current Input: ", self.chars)
            return None
        else:
            if self.check():
                while self.check():
                    char = self.getch()
                    if char == '\n' or char == '\r':
                        res = self.chars.strip()
                        self.chars = ""
                        return res
                    else:
                        self.chars += char
                        print("Current Input: ", self.chars)
            return None

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
    
    # Load action_dim and action_dtype from config file if not provided via command line
    if not hasattr(args, 'action_dim') or args.action_dim == 7:  # Default value
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            if 'action_dim' in config_data:
                args.action_dim = config_data['action_dim']
                print(f"Using action_dim from config: {args.action_dim}")
        except Exception as e:
            print(f"Could not read action_dim from config: {e}")
    
    if not hasattr(args, 'action_dtype') or args.action_dtype == 'float64':  # Default value
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            if 'action_dtype' in config_data:
                args.action_dtype = config_data['action_dtype']
                print(f"Using action_dtype from config: {args.action_dtype}")
        except Exception as e:
            print(f"Could not read action_dtype from config: {e}")
    
    return args

def make_robot(robot_cfg: Dict, args, max_connect_retry: int = 5):
    full_path = robot_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    RobotCls = getattr(module, class_name)
    print(f"Creating robot: {full_path}")
    robot = RobotCls(
        extra_args=args,
        **{k: v for k, v in robot_cfg.items() if k != 'target'}
    )
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot

class RobotController:
    def __init__(self, robot, shm_name, shm_shape, shm_dtype, robot_config_path):
        self.robot = robot
        self.robot_config_path = robot_config_path
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.shm = None
        self.action_buffer = None
        self.last_timestamp = 0.0
        self.stop_event = mp.Event()
    
    def is_connected(self):
        return self.shm is not None

    def connect_to_buffer(self, max_retries=30, retry_delay=1.0):
        """
        Connect to shared memory with retries to handle cases where the controller hasn't started yet.
        """
        for attempt in range(max_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                self.action_buffer = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
                print("Robot controller: Successfully connected to shared memory.")
                return
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    print(f"Robot controller: Shared memory '{self.shm_name}' not found, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Robot controller: Error! Shared memory '{self.shm_name}' does not exist after {max_retries} attempts.")
                    raise
   
    def run(self):
        try:
            # Create a new robot instance in this process since PyBullet connections don't work across processes
            print("Creating new robot instance in background process...")
            # Import robot configuration
            import yaml
            with open(self.robot_config_path, 'r') as f:
                robot_cfg = yaml.safe_load(f)
            
            # Create new robot instance
            full_path = robot_cfg['target']
            module_path, class_name = full_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            RobotCls = getattr(module, class_name)
            
            # Use consistent format: pass all parameters except 'target', and force use_gui=True
            robot_params = {k: v for k, v in robot_cfg.items() if k != 'target'}
            robot_params['use_gui'] = True  # Enable GUI for visualization
            self.robot = RobotCls(**robot_params)
            
            if not self.robot.connect():
                print("Failed to connect robot in background process")
                return
            
            self.connect_to_buffer()
            while not self.stop_event.is_set():
                # print("Try to get action from buffer...")
                current_timestamp = self.action_buffer[0]['timestamp']
                # print(current_timestamp, self.last_timestamp)
                if current_timestamp > self.last_timestamp:
                    self.last_timestamp = current_timestamp
                    action = self.action_buffer[0]['action'].copy()
                    self.robot.publish_action(action)
        except Exception as e:
            print(f"Robot controller: Error in run loop: {e}")
            print("Robot controller: Continuing without shared memory connection...")
        finally:
            print("\nRobot controller: Shutting down...")
            if self.shm:
                # Explicitly prevent resource tracker cleanup
                try:
                    import multiprocessing.resource_tracker
                    if hasattr(multiprocessing.resource_tracker._resource_tracker, 'unregister'):
                        multiprocessing.resource_tracker._resource_tracker.unregister(self.shm._name, 'shared_memory')
                except:
                    pass
                
                self.shm.close()
                # Note: We only close() the connection, we do NOT unlink() the shared memory
                # The shared memory is managed by start_teleop_controller.py

    def stop(self):
        self.stop_event.set()

def save_episode_to_hdf5(save_dir, episode_id, observations, actions):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'episode_{episode_id:04d}.hdf5')
    def write_group(group, data_list, key_prefix=None):
        # data_list: list of dict or value
        if isinstance(data_list[0], dict):
            # For each key, collect list of values and recurse
            for key in data_list[0].keys():
                sub_list = [obs[key] for obs in data_list]
                if isinstance(sub_list[0], dict):
                    sub_group = group.create_group(key)
                    write_group(sub_group, sub_list)
                else:
                    try:
                        group.create_dataset(key, data=np.stack(sub_list))
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Could not stack data for key '{key}'. Skipping. Error: {e}")
        else:
            # If not dict, just create dataset
            try:
                if key_prefix is None:
                    group.create_dataset('data', data=np.stack(data_list))
                else:
                    group.create_dataset(key_prefix, data=np.stack(data_list))
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        obs_group = f.create_group('observations')
        if observations:
            write_group(obs_group, observations)

# Global event to signal all threads to shut down gracefully.
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    if not shutdown_event.is_set():
        print("\nCtrl+C detected! Shutting down gracefully...", flush=True)
        print("Note: This will only close the recorder process, not the shared memory.", flush=True)
        shutdown_event.set()

# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...# ...existing code...
# ...existing code...

def robot_controller_run(robot, shm_name, shm_shape, shm_dtype, stop_event):
    """
    Standalone function to run the robot controller loop.
    """
    import numpy as np
    from multiprocessing import shared_memory
    import time

    shm = None
    action_buffer = None
    last_timestamp = 0.0

    # Retry connection to shared memory
    max_retries = 30
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            action_buffer = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
            print("Robot controller: Successfully connected to shared memory.", flush=True)
            break
        except FileNotFoundError:
            if attempt < max_retries - 1:
                print(f"Robot controller: Shared memory '{shm_name}' not found, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"Robot controller: Error! Shared memory '{shm_name}' does not exist after {max_retries} attempts.", flush=True)
                return

    try:
        while not stop_event.is_set():
            print("Try to get action from buffer...", flush=True)
            current_timestamp = action_buffer[0]['timestamp']
            print(current_timestamp, last_timestamp, flush=True)
            if current_timestamp > last_timestamp:
                last_timestamp = current_timestamp
                action = action_buffer[0]['action'].copy()
                robot.publish_action(action)
    except Exception as e:
        print("Exception in robot_controller_run:", e, flush=True)
        import traceback; traceback.print_exc()
    finally:
        print("\nRobot controller: Shutting down...", flush=True)
        if shm:
            # Explicitly prevent resource tracker cleanup
            try:
                import multiprocessing.resource_tracker
                if hasattr(multiprocessing.resource_tracker._resource_tracker, 'unregister'):
                    multiprocessing.resource_tracker._resource_tracker.unregister(shm._name, 'shared_memory')
            except:
                pass
            
            shm.close()
            # Note: We only close() the connection, we do NOT unlink() the shared memory
            # The shared memory is managed by start_teleop_controller.py

if __name__ == '__main__':
    # Use spawn method to avoid resource tracker issues
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    set_seed(0)
    args = parse_param()
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize non-blocking keyboard input
    kb_hit = KBHit()
    kb_hit.set_curses_term()

    # --- 1. Create Real-World Environment ---
    print(f"Loading robot configuration from {args.config}")
    with open(args.config, 'r') as f:
        robot_cfg = yaml.safe_load(f)
    
    # Force no GUI for main process robot (background process will have GUI)
    robot_cfg['use_gui'] = False
    
    robot = make_robot(robot_cfg, args)
    print("Robot successfully loaded (no GUI - background process will show GUI).")
    
    robot_controller = None
    controller_worker = None
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
            robot_config_path=args.config,
        )
        if os.name == 'nt':
            # Windows: use thread
            controller_worker = threading.Thread(target=robot_controller.run, daemon=True)
        else:
            # Linux/macOS: use process
            controller_worker = mp.Process(target=robot_controller.run)
        controller_worker.start()
        print("Robot controller worker started in the background.")
        time.sleep(1)
        action_buffer = None
        action_shm = None
        # Try to connect to shared memory for logging (with retries)
        max_retries = 10
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                action_shm = shared_memory.SharedMemory(name=shm_info['name'])
                action_buffer = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=action_shm.buf)
                print("Main process connected to shared memory for logging.")
                break
            except (FileNotFoundError, TypeError):
                if attempt < max_retries - 1:
                    print(f"Main process: Shared memory not found for logging, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print("Warning: Could not connect to shared memory for logging. Actions will not be saved.")
                    action_shm = None
                    action_buffer = None
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
                    if hasattr(robot, 'save_episode'):
                        robot.save_episode(os.path.join(args.save_dir, f'episode_{episode_count:04d}.hdf5'), observations, actions)
                    else:
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
        print("[Main Process] Note: Shared memory will remain available for other processes.")
        shutdown_event.set()
        kb_hit.set_normal_term() # Restore terminal settings

        if robot_controller:
            robot_controller.stop()
            print("Stop signal sent to robot controller.")

        if controller_worker is not None:
            if os.name == 'nt':
                controller_worker.join(timeout=2)
                if controller_worker.is_alive():
                    print("Controller thread did not terminate gracefully.")
                else:
                    print("Robot controller thread joined successfully.")
            else:
                controller_worker.join(timeout=2)
                if controller_worker.is_alive():
                    print("Controller process did not terminate gracefully, forcing termination.")
                    controller_worker.terminate()
                else:
                    print("Robot controller process joined successfully.")

        if action_shm:
            print("Closing shared memory connection (NOT destroying the memory block)...")
            action_shm.close()
            print("Main process shared memory link closed.")
            print("Shared memory block remains available for other processes.")
            # Note: We only close() the connection, we do NOT unlink() the shared memory
            # The shared memory is managed by start_teleop_controller.py

        if robot:
            robot.shutdown()
            print("Robot shutdown command sent.")
            
        print("Cleanup complete. Exiting.")