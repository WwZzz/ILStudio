"""
Base Visualizer for Robot Data Visualization

This module provides the base class for visualizers that can display robot data
from shared memory. Each robot module can provide its own Visualizer class that
inherits from BaseVisualizer.

Usage:
    1. Create a Visualizer class in your robot module (e.g., deploy/robot/so101_sim/visualizer.py)
    2. Export it in the module's __init__.py
    3. collect_data.py will automatically detect and start the visualizer as a subprocess
"""

import time
import importlib
from abc import ABC, abstractmethod
from typing import Optional

from deploy.shm_utils import SharedMemoryChannel


class BaseVisualizer(ABC):
    """
    Base class for robot visualizers.
    
    Visualizers read robot data from shared memory and display it.
    Each visualizer runs as a separate subprocess to avoid blocking the main process.
    """
    
    def __init__(self, shm_name: str, fps: float = 60.0, **kwargs):
        """
        Initialize the visualizer.
        
        Args:
            shm_name: Name of the shared memory to read robot data from
            fps: Target visualization frame rate (default: 60)
        """
        self.shm_name = shm_name
        self.fps = fps
        self.shm: Optional[SharedMemoryChannel] = None
        self.is_running = False

    def connect(self, timeout: float = 30.0) -> bool:
        """
        Connect to the robot's shared memory.
        
        Args:
            timeout: Maximum time to wait for SHM to be available
            
        Returns:
            True if connected successfully
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.shm = SharedMemoryChannel(self.shm_name, is_writer=False, timeout=1.0)
                print(f"[{self.__class__.__name__}] Connected to SHM: {self.shm_name}")
                return True
            except Exception as e:
                print(f"[{self.__class__.__name__}] Waiting for SHM '{self.shm_name}'... ({e})")
                time.sleep(0.5)
        
        print(f"[{self.__class__.__name__}] Failed to connect to SHM: {self.shm_name}")
        return False
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the visualization environment (e.g., create window, load model).
        Called once after connecting to SHM.
        
        Returns:
            True if setup successful
        """
        pass
    
    @abstractmethod
    def visualize(self, data: dict) -> bool:
        """
        Visualize one frame of data.
        
        Args:
            data: Data dictionary from shared memory (typically contains 'qpos', 'gpos', etc.)
            
        Returns:
            True to continue, False to stop
        """
        pass
    
    def cleanup(self):
        """
        Cleanup resources (e.g., close windows).
        Called when stopping.
        """
        pass

    def start(self):
        """
        Main loop: connect to SHM, setup, and continuously visualize data.
        """
        if not self.connect():
            return
        
        if not self.setup():
            print(f"[{self.__class__.__name__}] Setup failed")
            return
        
        self.is_running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0
        
        print(f"[{self.__class__.__name__}] Visualization started at {self.fps} FPS")
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                last_frame_time = current_time
                
                # Read data from SHM
                data = self.shm.read(blocking=False, skip_unchanged=False)
                if data is not None:
                    if not self.visualize(data):
                        break
                        
        except KeyboardInterrupt:
            print(f"\n[{self.__class__.__name__}] Interrupted")
        finally:
            self.cleanup()
            if self.shm:
                try:
                    self.shm.destroy()
                except Exception:
                    pass
            print(f"[{self.__class__.__name__}] Stopped")
    
    def stop(self):
        """Stop the visualization loop."""
        self.is_running = False


def start_visualizer(visualizer_type: str, shm_name: str, **kwargs):
    """
    Start a visualizer by its type string.
    
    This function is called by collect_data.py to start visualizers in subprocesses.
    
    Args:
        visualizer_type: Full module path to the Visualizer class
                        (e.g., "deploy.robot.so101_sim.Visualizer")
        shm_name: Name of the shared memory to visualize
        **kwargs: Additional arguments passed to the Visualizer constructor
    """
    try:
        # Import the visualizer class
        parts = visualizer_type.rsplit('.', 1)
        module_path = parts[0]
        class_name = parts[1] if len(parts) > 1 else "Visualizer"
        
        module = importlib.import_module(module_path)
        visualizer_class = getattr(module, class_name)
        
        # Create and start
        visualizer = visualizer_class(shm_name=shm_name, **kwargs)
        visualizer.start()
        
    except Exception as e:
        print(f"[start_visualizer] Error: {e}")
        import traceback
        traceback.print_exc()


def get_visualizer_class(device_type: str) -> Optional[type]:
    """
    Check if a device module has a Visualizer class.
    
    Args:
        device_type: Full module path to the device class
                    (e.g., "deploy.robot.so101_sim.So101SimRobot")
    
    Returns:
        The Visualizer class if found, None otherwise
    """
    try:
        # Get the module path (without the class name)
        parts = device_type.rsplit('.', 1)
        module_path = parts[0]
        
        # Try to import the module and check for Visualizer
        module = importlib.import_module(module_path)
        
        if hasattr(module, 'Visualizer'):
            visualizer_class = getattr(module, 'Visualizer')
            # Check if it's a class (could be BaseVisualizer subclass or CameraVisualizer)
            if isinstance(visualizer_class, type):
                return visualizer_class
        
        return None
        
    except Exception:
        return None


def get_visualizer_type_string(device_type: str) -> Optional[str]:
    """
    Get the full type string for a device's Visualizer class.
    
    Args:
        device_type: Full module path to the device class
        
    Returns:
        Full module path to Visualizer class, or None if not found
    """
    viz_class = get_visualizer_class(device_type)
    if viz_class is not None:
        parts = device_type.rsplit('.', 1)
        return parts[0] + ".Visualizer"
    return None


def is_mujoco_device_type(device_type: str) -> bool:
    try:
        parts = device_type.rsplit('.', 1)
        module_path = parts[0]
        class_name = parts[1]
        module = importlib.import_module(module_path)
        device_class = getattr(module, class_name)
        from deploy.simulation.mujoco import MujocoDeviceBase

        return issubclass(device_class, MujocoDeviceBase)
    except Exception:
        return False


def start_all_visualizers(
    device_configs: list,
    get_shm_name_func,
    is_camera_config_func,
    camera_fps: float = 30.0,
    camera_scale: float = 0.5,
) -> list:
    """
    Start visualizers for all devices that support visualization.
    
    This function:
    1. Collects all camera devices and starts a single CameraVisualizer for them
    2. Starts individual visualizers for other devices that have a Visualizer class
    
    Args:
        device_configs: List of device configuration dicts
        get_shm_name_func: Function to get SHM name from config
        is_camera_config_func: Function to check if config is a camera
        camera_fps: FPS for camera visualizer
        camera_scale: Scale factor for camera display
        
    Returns:
        List of started Process objects
    """
    import multiprocessing as mp
    from loguru import logger
    
    viz_procs = []
    camera_shm_names = []
    
    for cfg in device_configs:
        device_type = cfg.get("type", "")
        device_shm_name = get_shm_name_func(cfg)
        device_args = cfg.get("args", {})

        # Visualizer devices are standalone processes already started by start_devices().
        # Do not recursively spawn a visualizer for a visualizer.
        if device_type.startswith("deploy.visualizer."):
            continue
        
        # Check if this is a camera device
        try:
            if is_camera_config_func(cfg):
                camera_shm_names.append(device_shm_name)
                continue
        except Exception:
            pass

        if is_mujoco_device_type(device_type):
            if not device_args.get(
                "enable_viewer", device_args.get("enable_viewer_proxy", True)
            ):
                continue
            from deploy.simulation.mujoco import (
                get_mujoco_viewer_command_shm_name,
                get_mujoco_viewer_state_shm_name,
            )

            viz_type = "deploy.visualizer.mujoco_proxy_visualizer.MujocoProxyVisualizer"
            default_camera = device_args.get("viewer_default_camera")
            viz_kwargs = {
                "robot_shm_name": device_shm_name,
                "viewer_command_shm_name": device_args.get("viewer_command_shm_name")
                or get_mujoco_viewer_command_shm_name(device_shm_name),
                "viewer_state_shm_name": device_args.get("viewer_state_shm_name")
                or get_mujoco_viewer_state_shm_name(device_shm_name),
                "default_camera_name": default_camera,
                "auto_open": True,
                "show_tcp_frame": device_args.get("viewer_show_tcp_frame", True),
                "fps": float(device_args.get("viewer_proxy_fps", 15.0)),
                "window_name": f"{device_shm_name} Viewer Proxy",
                "show_window": bool(device_args.get("viewer_proxy_show_window", False)),
            }
            logger.info("Starting MuJoCo proxy visualizer for {}: {}", device_shm_name, viz_type)
            viz_proc = mp.Process(
                target=start_visualizer,
                args=(viz_type, device_shm_name),
                kwargs=viz_kwargs,
                daemon=True
            )
            viz_proc.start()
            viz_procs.append(viz_proc)
            logger.info("MuJoCo proxy visualizer started for SHM: {}", device_shm_name)
            continue
        
        # Check if device module has its own Visualizer
        viz_class = get_visualizer_class(device_type)
        if viz_class is not None:
            parts = device_type.rsplit('.', 1)
            viz_type = parts[0] + ".Visualizer"
            
            logger.info("Starting visualizer for {}: {}", device_shm_name, viz_type)
            viz_kwargs = {
                key: device_args[key]
                for key in ("xml_path", "scene_name", "scene_xml_path")
                if device_args.get(key) is not None
            }
            viz_proc = mp.Process(
                target=start_visualizer,
                args=(viz_type, device_shm_name),
                kwargs=viz_kwargs,
                daemon=True
            )
            viz_proc.start()
            viz_procs.append(viz_proc)
            logger.info("Visualizer subprocess started for SHM: {}", device_shm_name)

    # Start a single camera visualizer for all cameras
    if camera_shm_names:
        from deploy.visualizer.camera_visualizer import start_camera_visualizer
        
        logger.info("Starting camera visualizer for: {}", camera_shm_names)
        viz_proc = mp.Process(
            target=start_camera_visualizer,
            args=(camera_shm_names,),
            kwargs={"fps": camera_fps, "scale": camera_scale},
            daemon=True
        )
        viz_proc.start()
        viz_procs.append(viz_proc)
        logger.info("Camera visualizer started for {} cameras", len(camera_shm_names))
    
    return viz_procs


def stop_all_visualizers(viz_procs: list):
    """
    Stop all visualizer processes.
    
    Args:
        viz_procs: List of Process objects to stop
    """
    for viz_proc in viz_procs:
        if viz_proc is not None:
            viz_proc.terminate()
            viz_proc.join(timeout=1.0)
            if viz_proc.is_alive():
                viz_proc.kill()