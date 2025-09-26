# deploy/controller.py
"""
Controller utilities for teleoperation recording and keyboard input handling.
"""

import os
import sys
import time
import yaml
import importlib
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

# Platform-specific imports
if os.name == 'nt':
    import msvcrt  # Windows only
else:
    import select  # Unix only
    import termios # Unix only


def infer_action_params_from_shm(shm_name: str):
    """
    Read action_dim and action_dtype from shared memory metadata.
    
    Args:
        shm_name: Name of the shared memory to inspect
        
    Returns:
        tuple: (action_dim, action_dtype) read from shared memory metadata
    """
    try:
        # Connect to existing shared memory
        shm = shared_memory.SharedMemory(name=shm_name)
        
        # Create a numpy array from the shared memory buffer using the new structured dtype
        # The new format includes metadata fields
        structured_dtype = np.dtype([
            ('action_dim', np.int32),      # Metadata: action dimension
            ('action_dtype_code', np.int32), # Metadata: dtype code
            ('timestamp', np.float64),
            ('action', (np.float64, 1)),  # Placeholder - we'll get the real size from metadata
        ])
        
        # First, read the metadata to get the actual action_dim and dtype_code
        shm_array = np.ndarray((1,), dtype=structured_dtype, buffer=shm.buf)
        action_dim = int(shm_array['action_dim'][0])
        dtype_code = int(shm_array['action_dtype_code'][0])
        
        # Convert dtype code back to numpy dtype
        from deploy.teleoperator.base import code2dtype
        action_dtype = code2dtype(dtype_code)
        
        shm.close()
        return action_dim, action_dtype
            
    except Exception as e:
        print(f"Warning: Could not read action parameters from shared memory '{shm_name}': {e}")
        # Return default values
        return 7, np.float64


def robot_controller_run(shm_name, shm_shape, shm_dtype, robot_config_path):
    """
    Standalone function to run the robot controller loop.
    """
    import numpy as np
    from multiprocessing import shared_memory
    import time
    import yaml
    import importlib
    import signal
    import threading

    # Create stop event for this process
    stop_event = threading.Event()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nRobot controller: Shutdown signal received...", flush=True)
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create robot instance
    print("Creating new robot instance in background process...")
    try:
        # Import robot configuration
        with open(robot_config_path, 'r') as f:
            robot_cfg = yaml.safe_load(f)
        
        # Create new robot instance
        full_path = robot_cfg['target']
        module_path, class_name = full_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        RobotCls = getattr(module, class_name)
        
        # Use consistent format: pass all parameters except 'target', and force use_gui=True
        robot_params = {k: v for k, v in robot_cfg.items() if k != 'target'}
        robot_params['use_gui'] = True  # Enable GUI for visualization
        robot = RobotCls(**robot_params)
        
        if not robot.connect():
            print("Failed to connect robot in background process")
            return
    except Exception as e:
        print(f"Failed to create robot in background process: {e}")
        import traceback
        traceback.print_exc()
        return

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
            # print("Try to get action from buffer...", flush=True)
            current_timestamp = action_buffer[0]['timestamp']
            # print(current_timestamp, last_timestamp, flush=True)
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


def setup_action_buffer(robot, args):
    """
    Setup action buffer and robot controller for shared memory communication.
    
    Args:
        robot: Robot instance
        args: Parsed command line arguments
        
    Returns:
        tuple: (action_buffer, action_shm) - action buffer and shared memory object
    """
    import os
    import threading
    import multiprocessing as mp
    from multiprocessing import shared_memory
    import numpy as np
    from deploy.teleoperator.base import generate_shm_info
    
    if args.shm_name:
        # Infer action parameters from existing shared memory
        action_dim, action_dtype = infer_action_params_from_shm(args.shm_name)
        print(f"Inferred action_dim: {action_dim}, action_dtype: {action_dtype}")
        
        shm_info = generate_shm_info(args.shm_name, action_dim, action_dtype)
        robot_controller = RobotController(
            shm_name=shm_info['name'],
            shm_shape=shm_info['shape'],
            shm_dtype=shm_info['dtype'],
            robot_config_path=args.config,
        )
        if os.name == 'nt':
            # Windows: use thread
            controller_worker = threading.Thread(target=robot_controller.run, daemon=True)
        else:
            # Linux/macOS: use process - pass parameters instead of the controller object
            controller_worker = mp.Process(
                target=robot_controller_run,
                args=(shm_info['name'], shm_info['shape'], shm_info['dtype'], args.config)
            )
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
        action_shm = None
    
    return action_buffer, action_shm


class KBHit:
    """Cross-platform keyboard input handler for non-blocking input."""
    
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
        """Restore normal terminal settings."""
        if os.name != 'nt':
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def set_curses_term(self):
        """Set terminal to curses mode for non-blocking input."""
        if os.name != 'nt':
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

    def check(self):
        """Check if input is available without blocking."""
        if os.name == 'nt':
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def getch(self):
        """Get a single character from input."""
        if os.name == 'nt':
            return msvcrt.getwch()
        else:
            return sys.stdin.read(1)

    def getarrow(self):
        """Get arrow key input (for future use)."""
        c1 = self.getch()
        if c1 == '\x1b': # ESC
            c2 = self.getch()
            c3 = self.getch()
            return c3
        return None

    def get_input(self):
        """Get line input from user."""
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


class RobotController:
    """Controller for managing robot actions from shared memory."""
    
    def __init__(self, shm_name, shm_shape, shm_dtype, robot_config_path):
        """
        Initialize the robot controller.
        
        Args:
            shm_name: Name of shared memory for action data
            shm_shape: Shape of shared memory buffer
            shm_dtype: Data type of shared memory buffer
            robot_config_path: Path to robot configuration file
        """
        # Don't store the robot instance as it may not be pickleable
        self.robot_config_path = robot_config_path
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.shm_dtype = shm_dtype
        self.shm = None
        self.action_buffer = None
        self.last_timestamp = 0.0
        self.stop_event = mp.Event()
    
    def is_connected(self):
        """Check if connected to shared memory."""
        return self.shm is not None

    def connect_to_buffer(self, max_retries=30, retry_delay=1.0):
        """
        Connect to shared memory with retries to handle cases where the controller hasn't started yet.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
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
        """Main control loop for the robot controller."""
        try:
            # Create a new robot instance in this process since PyBullet connections don't work across processes
            print("Creating new robot instance in background process...")
            # Import robot configuration
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
        """Stop the robot controller."""
        self.stop_event.set()
