# start_teleop_recorder.py (simplified argparse version)
import yaml
import os
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
from data_utils.utils import set_seed
from deploy.robot.base import AbstractRobotInterface, RateLimiter, make_robot
from deploy.teleoperator.base import generate_shm_info
from deploy.controller import KBHit, RobotController, infer_action_params_from_shm, robot_controller_run, setup_action_buffer

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    if not shutdown_event.is_set():
        print("\nCtrl+C detected! Shutting down gracefully...", flush=True)
        print("Note: This will only close the recorder process, not the shared memory.", flush=True)
        shutdown_event.set()

def parse_param():
    """Parse command line arguments for teleop recorder."""
    parser = argparse.ArgumentParser(description='Teleoperation data recorder')
    
    # Essential arguments
    parser.add_argument('--shm_name', type=str, default='ilstd_teleop_controller',
                       help='Name of shared memory for action data (optional)')
    parser.add_argument('--config', type=str, default='configs/robot/dummy.yaml',
                       help='Robot configuration file path')
    parser.add_argument('--frequency', type=int, default=100,
                       help='Recording frequency in Hz')
    parser.add_argument('--save_dir', type=str, default='data/teleop_recordings',
                       help='Directory to save recorded episodes')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting episode index')
    
    args = parser.parse_args()
    return args

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

if __name__ == '__main__':
    # Use spawn method to avoid resource tracker issues
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
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

    # --- 2. Setup action buffer and robot controller ---
    action_buffer, action_shm = setup_action_buffer(robot, args)
    
    # --- 3. Main Data Collection Loop ---
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