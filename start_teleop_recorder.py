# start_teleop_recorder_simple_multithread.py - Simplified multi-threaded teleoperation data recorder
import configs 
import os
import yaml
import time
import importlib
import numpy as np
import argparse
import signal
import sys
import h5py
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
import traceback
from loguru import logger
from deploy.robot.base import RateLimiter, make_robot
from deploy.teleoperator.base import generate_shm_info
from deploy.controller import KBHit, infer_action_params_from_shm

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    if not shutdown_event.is_set():
        logger.info("Ctrl+C detected! Shutting down gracefully...")
        logger.info("Note: This will only close the recorder process and its connection to shared memory.")
        logger.info("The shared memory block will remain available for other processes.")
        shutdown_event.set()

def parse_param():
    """Parse command line arguments for teleop recorder."""
    parser = argparse.ArgumentParser(description='Simple multi-threaded teleoperation data recorder')
    
    # Essential arguments
    parser.add_argument("-shm", '--shm_name', type=str, default='ilstd_teleop_controller',
                       help='Name of shared memory for action data (optional)')
    parser.add_argument("-c", '--config', type=str, default='robot/dummy',
                       help='Robot config (name under configs/robot or absolute path to yaml)')
    parser.add_argument("-f", '--frequency', type=int, default=25,
                       help='Recording frequency in Hz')
    parser.add_argument("-af", '--action_frequency', type=int, default=40,
                       help='Recording frequency in Hz')
    parser.add_argument("-of", '--observation_frequency', type=int, default=50,
                    help='Recording frequency in Hz')
    parser.add_argument("-o", '--output_dir', type=str, default='data/teleop_recordings',
                       help='Directory to save recorded episodes')
    parser.add_argument("-s", '--start_idx', type=int, default=0,
                       help='Starting episode index')
    
    args, unknown = parser.parse_known_args()
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
    args.unknown_overrides = cfg_loader._overrides
    return args

def save_episode_to_hdf5(output_dir, episode_id, observations, actions):
    """Save episode data to HDF5 file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'episode_{episode_id:04d}.hdf5')
    
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
                        logger.warning(f"Could not stack data for key '{key}'. Skipping. Error: {e}")
        else:
            # If not dict, just create dataset
            try:
                if key_prefix is None:
                    group.create_dataset('data', data=np.stack(data_list))
                else:
                    group.create_dataset(key_prefix, data=np.stack(data_list))
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not stack data for key '{key_prefix}'. Skipping. Error: {e}")

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        obs_group = f.create_group('observations')
        if observations:
            write_group(obs_group, observations)

class SimpleMultiThreadTeleopRecorder:
    """Simplified multi-threaded teleoperation recorder main class"""
    
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.action_buffer = None
        self.action_shm = None
        self.episode_count = args.start_idx
        
        # Thread control
        self.action_publisher_thread = None
        self.observation_collector_thread = None
        self.running = False
        
    def initialize_robot(self):
        """Initialize robot"""
        from configs.loader import ConfigLoader
        from configs.utils import apply_overrides_to_mapping
        from data_utils.utils import _convert_to_type
        try:
            cfg_path = ConfigLoader()._resolve('robot', self.args.config)
        except Exception:
            cfg_path = self.args.config
        logger.info(f"Loading robot configuration from {cfg_path}")
        with open(cfg_path, 'r') as f:
            robot_cfg = yaml.safe_load(f)
        # apply overrides passed via CLI
        apply_overrides_to_mapping(robot_cfg, self.args.unknown_overrides.get('robot', {}), _convert_to_type)
        
        # Force no GUI for main process robot
        robot_cfg['use_gui'] = False
        
        self.robot = make_robot(robot_cfg, self.args)
        logger.info("Robot successfully loaded.")
        
    def setup_action_buffer(self):
        """Setup action buffer"""
        if self.args.shm_name and self.args.shm_name.strip():
            # Infer action parameters
            action_dim, action_dtype = infer_action_params_from_shm(self.args.shm_name)
            logger.info(f"Inferred action_dim: {action_dim}, action_dtype: {action_dtype}")
            
            shm_info = generate_shm_info(self.args.shm_name, action_dim, action_dtype)
            
            # Directly connect to shared memory
            max_retries = 10
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    self.action_shm = shared_memory.SharedMemory(name=shm_info['name'])
                    self.action_buffer = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=self.action_shm.buf)
                    logger.info("Main process connected to shared memory.")
                    
                    # Verify shared memory accessibility
                    try:
                        _ = self.action_buffer[0]
                        logger.info("Shared memory is accessible and ready for use.")
                    except Exception as e:
                        logger.warning(f"Shared memory connected but not accessible: {e}")
                        self.action_shm = None
                        self.action_buffer = None
                    break
                except (FileNotFoundError, TypeError):
                    if attempt < max_retries - 1:
                        logger.info(f"Main process: Shared memory not found, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.warning("Could not connect to shared memory. Actions will not be saved.")
                        logger.warning("Make sure the controller process (start_teleop_controller.py) is running.")
                        self.action_shm = None
                        self.action_buffer = None
        else:
            logger.info("No shared memory name provided or empty. Action publishing will be disabled.")
            self.action_buffer = None
            self.action_shm = None
    
    def action_publisher_worker(self):
        """Action publisher worker thread"""
        # Check if action buffer exists, exit directly if not
        if self.action_buffer is None:
            logger.info("[ActionPublisher] No action buffer available, skipping action publishing thread")
            return
            
        logger.info(f"[ActionPublisher] Thread started, publishing at {self.args.action_frequency}Hz")
        rate_limiter = RateLimiter()
        last_timestamp = 0
        
        while self.running and not shutdown_event.is_set():
            try:
                if self.action_buffer is not None:
                    current_timestamp = self.action_buffer[0]['timestamp']
                    if current_timestamp > last_timestamp:
                        last_timestamp = current_timestamp
                        action = self.action_buffer[0]['action'].copy()
                        self.robot.publish_action(action)
                
                rate_limiter.sleep(self.args.action_frequency)
                
            except Exception as e:
                logger.error(f"[ActionPublisher] Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                
        logger.info("[ActionPublisher] Thread stopped")
    
    def observation_collector_worker(self):
        """Observation collector worker thread"""
        logger.info(f"[ObservationCollector] Thread started, collecting at {self.args.observation_frequency}Hz")
        rate_limiter = RateLimiter()
        
        while self.running and not shutdown_event.is_set():
            try:
                obs = self.robot.get_observation()
                if obs:
                    current_time = time.perf_counter()
                    obs['_timestamp'] = current_time
                    # Here we can store observation data to queue or process directly
                    # For simplicity, we temporarily don't process here
                
                rate_limiter.sleep(self.args.observation_frequency)
                
            except Exception as e:
                logger.error(f"[ObservationCollector] Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                
        logger.info("[ObservationCollector] Thread stopped")
    
    def start_threads(self):
        """Start background threads"""
        self.running = True
        
        # Start action publisher thread (only when action buffer exists)
        if self.action_buffer is not None:
            self.action_publisher_thread = threading.Thread(
                target=self.action_publisher_worker, 
                daemon=True
            )
            self.action_publisher_thread.start()
            logger.info("Action publisher thread started")
        else:
            logger.info("Action publisher thread skipped (no action buffer)")
        
        # Start observation collector thread
        self.observation_collector_thread = threading.Thread(
            target=self.observation_collector_worker, 
            daemon=True
        )
        self.observation_collector_thread.start()
        logger.info("Observation collector thread started")
        
        logger.info("Background threads started successfully")
    
    def stop_threads(self):
        """Stop background threads"""
        self.running = False
        
        if self.action_publisher_thread:
            self.action_publisher_thread.join(timeout=2)
            logger.info("Action publisher thread stopped")
            
        if self.observation_collector_thread:
            self.observation_collector_thread.join(timeout=2)
            logger.info("Observation collector thread stopped")
    
    def collect_episode_data(self, kb_hit):
        """Collect data for one episode"""
        logger.info(f"Starting episode {self.episode_count}. Recording...")
        
        observations, actions = [], []
        all_timestamps = []
        
        logger.info("Press Enter to STOP recording...")
        # Consume any prior input
        while kb_hit.get_input() is not None: 
            pass

        # Data collection loop
        stop_recording = False
        rate_limiter = RateLimiter()
        
        while not stop_recording and not shutdown_event.is_set():
            if kb_hit.get_input() is not None:
                stop_recording = True
            else:
                # Get observation data
                obs = self.robot.get_observation()
                if obs:
                    current_time = time.perf_counter()
                    obs['_timestamp'] = current_time
                    observations.append(obs)
                    all_timestamps.append(current_time)
                    
                    # Get action data
                    if self.action_buffer is not None:
                        action = self.action_buffer[0]['action'].copy()
                        actions.append(action)
                
                rate_limiter.sleep(self.args.frequency)

        if shutdown_event.is_set():
            return None, None
            
        if all_timestamps:
            actual_frequency = len(all_timestamps) / (all_timestamps[-1] - all_timestamps[0])
            logger.info(f"Episode {self.episode_count} finished at {actual_frequency:.2f}Hz ({self.args.frequency}Hz expected). Collected {len(observations)} timesteps.")
        else:
            logger.info(f"Episode {self.episode_count} finished. No data collected.")
            
        return observations, actions
    
    def save_episode(self, observations, actions):
        """Save episode data"""
        if observations:
            if hasattr(self.robot, 'save_episode'):
                self.robot.save_episode(
                    os.path.join(self.args.output_dir, f'episode_{self.episode_count:04d}.hdf5'), 
                    observations, 
                    actions
                )
            else:
                save_episode_to_hdf5(self.args.output_dir, self.episode_count, observations, actions)
            logger.info(f"Episode {self.episode_count} was successfully saved to {self.args.output_dir}.")
            self.episode_count += 1
        else:
            logger.info("No data collected, skipping save.")
    
    def run(self):
        """Main run loop"""
        # Initialize non-blocking keyboard input
        kb_hit = KBHit()
        kb_hit.set_curses_term()
        
        try:
            # Initialize robot
            self.initialize_robot()
            
            # Setup action buffer
            self.setup_action_buffer()
            
            # Start background threads
            self.start_threads()
            
            # Main data collection loop
            while not shutdown_event.is_set():
                # Wait for user to start episode
                logger.info(f"{'='*10}")
                logger.info(f"Press Enter to START episode {self.episode_count}...")
                logger.info(f"{'='*10}")
                while not shutdown_event.is_set():
                    if kb_hit.get_input() is not None:
                        break
                    time.sleep(0.1)
                
                if shutdown_event.is_set(): 
                    break

                # Collect episode data
                observations, actions = self.collect_episode_data(kb_hit)
                
                if shutdown_event.is_set(): 
                    break

                # Ask whether to save
                logger.info("Save this episode? (Press Enter to SAVE, or type anything and press Enter to DISCARD)")
                saving_prompt = None
                while saving_prompt is None and not shutdown_event.is_set():
                    saving_prompt = kb_hit.get_input()
                    if saving_prompt is None:
                        time.sleep(0.1)
                
                if shutdown_event.is_set(): 
                    break

                if len(saving_prompt) == 0:
                    self.save_episode(observations, actions)
                else:
                    logger.info("Discarding episode.")

        except KeyboardInterrupt:
            logger.info("[Main Process] Exit by KeyboardInterrupt (fallback).")
        finally:
            # Graceful shutdown
            logger.info("[Main Process] Shutting down...")
            logger.info("[Main Process] Note: Only closing recorder connection to shared memory.")
            logger.info("[Main Process] The shared memory block will remain available for other processes.")
            shutdown_event.set()
            kb_hit.set_normal_term()

            # Stop background threads
            self.stop_threads()

            # Close shared memory connection (do not destroy shared memory block)
            if self.action_shm:
                logger.info("Closing shared memory connection (NOT destroying the memory block)...")
                try:
                    # Check if shared memory is still valid
                    try:
                        # Try to access shared memory to check if it still exists
                        _ = self.action_buffer[0]
                        logger.info("Shared memory is still accessible before closing connection.")
                    except Exception as e:
                        logger.warning(f"Shared memory may have been closed by another process: {e}")
                    
                    # Prevent resource tracker from automatically cleaning up shared memory
                    try:
                        import multiprocessing.resource_tracker
                        if hasattr(multiprocessing.resource_tracker._resource_tracker, 'unregister'):
                            multiprocessing.resource_tracker._resource_tracker.unregister(self.action_shm._name, 'shared_memory')
                            logger.info("Successfully unregistered shared memory from resource tracker.")
                    except Exception as e:
                        logger.warning(f"Could not unregister from resource tracker: {e}")
                    
                    self.action_shm.close()  # Only close connection, do not destroy shared memory block
                    logger.info("Main process shared memory connection closed.")
                    logger.info("Shared memory block should remain available for other processes.")
                    logger.info("Note: If shared memory is no longer available, it may have been closed by the controller process.")
                except Exception as e:
                    logger.warning(f"Error closing shared memory connection: {e}")
                    logger.info("Shared memory block should still be available for other processes.")

            # Close robot
            if self.robot:
                self.robot.shutdown()
                logger.info("Robot shutdown command sent.")
                
            logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    # Use spawn method to avoid resource tracker issues
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_param()
    signal.signal(signal.SIGINT, signal_handler)

    # Create and run simplified multi-threaded recorder
    recorder = SimpleMultiThreadTeleopRecorder(args)
    recorder.run()
