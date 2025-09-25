# start_teleop_recorder_simple_multithread.py - 简化多线程版本的遥操作数据记录器
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
import traceback
from deploy.robot.base import RateLimiter, make_robot
from deploy.teleoperator.base import generate_shm_info
from deploy.controller import KBHit, infer_action_params_from_shm

# Global shutdown event
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    if not shutdown_event.is_set():
        print("\nCtrl+C detected! Shutting down gracefully...", flush=True)
        print("Note: This will only close the recorder process and its connection to shared memory.", flush=True)
        print("The shared memory block will remain available for other processes.", flush=True)
        shutdown_event.set()

def parse_param():
    """Parse command line arguments for teleop recorder."""
    parser = argparse.ArgumentParser(description='Simple multi-threaded teleoperation data recorder')
    
    # Essential arguments
    parser.add_argument('--shm_name', type=str, default='ilstd_teleop_controller',
                       help='Name of shared memory for action data (optional)')
    parser.add_argument('--config', type=str, default='robot/dummy',
                       help='Robot config (name under configs/robot or absolute path to yaml)')
    parser.add_argument('--frequency', "-freq", type=int, default=25,
                       help='Recording frequency in Hz')
    parser.add_argument('--action_frequency', "-afreq", type=int, default=40,
                       help='Recording frequency in Hz')
    parser.add_argument('--observation_frequency', "-ofreq", type=int, default=50,
                    help='Recording frequency in Hz')
    parser.add_argument('--save_dir', type=str, default='data/teleop_recordings',
                       help='Directory to save recorded episodes')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting episode index')
    
    args, unknown = parser.parse_known_args()
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
    args.unknown_overrides = cfg_loader._overrides
    return args

def save_episode_to_hdf5(save_dir, episode_id, observations, actions):
    """Save episode data to HDF5 file."""
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

class SimpleMultiThreadTeleopRecorder:
    """简化多线程遥操作记录器主类"""
    
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.action_buffer = None
        self.action_shm = None
        self.episode_count = args.start_idx
        
        # 线程控制
        self.action_publisher_thread = None
        self.observation_collector_thread = None
        self.running = False
        
    def initialize_robot(self):
        """初始化机器人"""
        from configs.loader import ConfigLoader
        from configs.utils import apply_overrides_to_mapping
        from data_utils.utils import _convert_to_type
        try:
            cfg_path = ConfigLoader()._resolve('robot', self.args.config)
        except Exception:
            cfg_path = self.args.config
        print(f"Loading robot configuration from {cfg_path}")
        with open(cfg_path, 'r') as f:
            robot_cfg = yaml.safe_load(f)
        # apply overrides passed via CLI
        apply_overrides_to_mapping(robot_cfg, self.args.unknown_overrides.get('robot', {}), _convert_to_type)
        
        # Force no GUI for main process robot
        robot_cfg['use_gui'] = False
        
        self.robot = make_robot(robot_cfg, self.args)
        print("Robot successfully loaded.")
        
    def setup_action_buffer(self):
        """设置动作缓冲区"""
        if self.args.shm_name and self.args.shm_name.strip():
            # 推断动作参数
            action_dim, action_dtype = infer_action_params_from_shm(self.args.shm_name)
            print(f"Inferred action_dim: {action_dim}, action_dtype: {action_dtype}")
            
            shm_info = generate_shm_info(self.args.shm_name, action_dim, action_dtype)
            
            # 直接连接共享内存
            max_retries = 10
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    self.action_shm = shared_memory.SharedMemory(name=shm_info['name'])
                    self.action_buffer = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=self.action_shm.buf)
                    print("Main process connected to shared memory.")
                    
                    # 验证共享内存是否可访问
                    try:
                        _ = self.action_buffer[0]
                        print("Shared memory is accessible and ready for use.")
                    except Exception as e:
                        print(f"Warning: Shared memory connected but not accessible: {e}")
                        self.action_shm = None
                        self.action_buffer = None
                    break
                except (FileNotFoundError, TypeError):
                    if attempt < max_retries - 1:
                        print(f"Main process: Shared memory not found, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        print("Warning: Could not connect to shared memory. Actions will not be saved.")
                        print("Make sure the controller process (start_teleop_controller.py) is running.")
                        self.action_shm = None
                        self.action_buffer = None
        else:
            print("No shared memory name provided or empty. Action publishing will be disabled.")
            self.action_buffer = None
            self.action_shm = None
    
    def action_publisher_worker(self):
        """动作发布工作线程"""
        # 检查是否有动作缓冲区，如果没有则直接退出
        if self.action_buffer is None:
            print("[ActionPublisher] No action buffer available, skipping action publishing thread")
            return
            
        print(f"[ActionPublisher] Thread started, publishing at {self.args.action_frequency}Hz")
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
                print(f"[ActionPublisher] Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                
        print("[ActionPublisher] Thread stopped")
    
    def observation_collector_worker(self):
        """观测收集工作线程"""
        print(f"[ObservationCollector] Thread started, collecting at {self.args.observation_frequency}Hz")
        rate_limiter = RateLimiter()
        
        while self.running and not shutdown_event.is_set():
            try:
                obs = self.robot.get_observation()
                if obs:
                    current_time = time.perf_counter()
                    obs['_timestamp'] = current_time
                    # 这里可以将观测数据存储到队列或直接处理
                    # 为了简化，我们暂时不在这里处理
                
                rate_limiter.sleep(self.args.observation_frequency)
                
            except Exception as e:
                print(f"[ObservationCollector] Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)
                
        print("[ObservationCollector] Thread stopped")
    
    def start_threads(self):
        """启动后台线程"""
        self.running = True
        
        # 启动动作发布线程（仅当有动作缓冲区时）
        if self.action_buffer is not None:
            self.action_publisher_thread = threading.Thread(
                target=self.action_publisher_worker, 
                daemon=True
            )
            self.action_publisher_thread.start()
            print("Action publisher thread started")
        else:
            print("Action publisher thread skipped (no action buffer)")
        
        # 启动观测收集线程
        self.observation_collector_thread = threading.Thread(
            target=self.observation_collector_worker, 
            daemon=True
        )
        self.observation_collector_thread.start()
        print("Observation collector thread started")
        
        print("Background threads started successfully")
    
    def stop_threads(self):
        """停止后台线程"""
        self.running = False
        
        if self.action_publisher_thread:
            self.action_publisher_thread.join(timeout=2)
            print("Action publisher thread stopped")
            
        if self.observation_collector_thread:
            self.observation_collector_thread.join(timeout=2)
            print("Observation collector thread stopped")
    
    def collect_episode_data(self, kb_hit):
        """收集一个episode的数据"""
        print(f"Starting episode {self.episode_count}. Recording...")
        
        observations, actions = [], []
        all_timestamps = []
        
        print("Press Enter to STOP recording...")
        # Consume any prior input
        while kb_hit.get_input() is not None: 
            pass

        # 数据收集循环
        stop_recording = False
        rate_limiter = RateLimiter()
        
        while not stop_recording and not shutdown_event.is_set():
            if kb_hit.get_input() is not None:
                stop_recording = True
            else:
                # 获取观测数据
                obs = self.robot.get_observation()
                if obs:
                    current_time = time.perf_counter()
                    obs['_timestamp'] = current_time
                    observations.append(obs)
                    all_timestamps.append(current_time)
                    
                    # 获取动作数据
                    if self.action_buffer is not None:
                        action = self.action_buffer[0]['action'].copy()
                        actions.append(action)
                
                rate_limiter.sleep(self.args.frequency)

        if shutdown_event.is_set():
            return None, None
            
        if all_timestamps:
            actual_frequency = len(all_timestamps) / (all_timestamps[-1] - all_timestamps[0])
            print(f"Episode {self.episode_count} finished at {actual_frequency:.2f}Hz ({self.args.frequency}Hz expected). Collected {len(observations)} timesteps.")
        else:
            print(f"Episode {self.episode_count} finished. No data collected.")
            
        return observations, actions
    
    def save_episode(self, observations, actions):
        """保存episode数据"""
        if observations:
            if hasattr(self.robot, 'save_episode'):
                self.robot.save_episode(
                    os.path.join(self.args.save_dir, f'episode_{self.episode_count:04d}.hdf5'), 
                    observations, 
                    actions
                )
            else:
                save_episode_to_hdf5(self.args.save_dir, self.episode_count, observations, actions)
            print(f"Episode {self.episode_count} was successfully saved to {self.args.save_dir}.")
            self.episode_count += 1
        else:
            print("No data collected, skipping save.")
    
    def run(self):
        """主运行循环"""
        # 初始化非阻塞键盘输入
        kb_hit = KBHit()
        kb_hit.set_curses_term()
        
        try:
            # 初始化机器人
            self.initialize_robot()
            
            # 设置动作缓冲区
            self.setup_action_buffer()
            
            # 启动后台线程
            self.start_threads()
            
            # 主数据收集循环
            while not shutdown_event.is_set():
                # 等待用户开始episode
                print(f"\n{'='*10}\nPress Enter to START episode {self.episode_count}...\n{'='*10}")
                while not shutdown_event.is_set():
                    if kb_hit.get_input() is not None:
                        break
                    time.sleep(0.1)
                
                if shutdown_event.is_set(): 
                    break

                # 收集episode数据
                observations, actions = self.collect_episode_data(kb_hit)
                
                if shutdown_event.is_set(): 
                    break

                # 询问是否保存
                print("Save this episode? (Press Enter to SAVE, or type anything and press Enter to DISCARD)")
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
                    print("Discarding episode.")

        except KeyboardInterrupt:
            print("\n[Main Process] Exit by KeyboardInterrupt (fallback).")
        finally:
            # 优雅关闭
            print("\n[Main Process] Shutting down...")
            print("[Main Process] Note: Only closing recorder connection to shared memory.")
            print("[Main Process] The shared memory block will remain available for other processes.")
            shutdown_event.set()
            kb_hit.set_normal_term()

            # 停止后台线程
            self.stop_threads()

            # 关闭共享内存连接（不销毁共享内存块）
            if self.action_shm:
                print("Closing shared memory connection (NOT destroying the memory block)...")
                try:
                    # 检查共享内存是否仍然有效
                    try:
                        # 尝试访问共享内存来检查它是否仍然存在
                        _ = self.action_buffer[0]
                        print("Shared memory is still accessible before closing connection.")
                    except Exception as e:
                        print(f"Warning: Shared memory may have been closed by another process: {e}")
                    
                    # 防止资源跟踪器自动清理共享内存
                    try:
                        import multiprocessing.resource_tracker
                        if hasattr(multiprocessing.resource_tracker._resource_tracker, 'unregister'):
                            multiprocessing.resource_tracker._resource_tracker.unregister(self.action_shm._name, 'shared_memory')
                            print("Successfully unregistered shared memory from resource tracker.")
                    except Exception as e:
                        print(f"Warning: Could not unregister from resource tracker: {e}")
                    
                    self.action_shm.close()  # 只关闭连接，不销毁共享内存块
                    print("Main process shared memory connection closed.")
                    print("Shared memory block should remain available for other processes.")
                    print("Note: If shared memory is no longer available, it may have been closed by the controller process.")
                except Exception as e:
                    print(f"Warning: Error closing shared memory connection: {e}")
                    print("Shared memory block should still be available for other processes.")

            # 关闭机器人
            if self.robot:
                self.robot.shutdown()
                print("Robot shutdown command sent.")
                
            print("Cleanup complete. Exiting.")

if __name__ == '__main__':
    # Use spawn method to avoid resource tracker issues
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_param()
    signal.signal(signal.SIGINT, signal_handler)

    # 创建并运行简化多线程记录器
    recorder = SimpleMultiThreadTeleopRecorder(args)
    recorder.run()
