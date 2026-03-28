from abc import ABC, abstractmethod
from deploy.utils import RateLimiter
# Import shm_utils early to apply resource_tracker patch before any subprocess spawns
from deploy.shm_utils import SharedMemoryChannel, _fix_resource_tracker
from loguru import logger
import importlib
import numpy as np
import signal
import sys
from benchmark.base import MetaObs


def start_device(device_config:dict):
    """
    Start a device in a subprocess.
    Handles SIGTERM/SIGINT to ensure graceful cleanup of device resources (e.g., serial ports).
    """
    # Re-apply resource_tracker patch in subprocess (critical: must be done early)
    _fix_resource_tracker()
    
    device_type = device_config['type']
    parts = device_type.rsplit('.', 1)
    device_module_name = parts[0]
    device_class_name = parts[1]
    device_module = importlib.import_module(device_module_name)
    device_class = getattr(device_module, device_class_name)
    device = device_class(**device_config['args'])
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"[{device.name}] Received signal {signum}, stopping device...")
        device.is_running = False
        # Call close() to cleanup resources (shm, serial ports, etc.)
        if hasattr(device, 'close'):
            device.close()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        device.start()
    except Exception as e:
        logger.error(f"[{device.name}] Error in device.start(): {e}")
    finally:
        # Ensure cleanup even if start() exits unexpectedly
        device.is_running = False
        if hasattr(device, 'close'):
            device.close()
    
    return device

class BaseDevice(ABC):
    def __init__(self, name:str, max_size_mb:int=64, fps:float=1000):
        self.name = name
        self.max_size_mb = max_size_mb
        self.shm = None
        self.is_running = False
        self.fps = fps

    def connect_to_existing_shm(self, shm_name:str):
        """
        Connect to the shared memory
        """
        try:
            shm = SharedMemoryChannel(shm_name, is_writer=False)
            return shm
        except FileNotFoundError:
            raise ValueError(f"Shared memory '{shm_name}' not found")
    
    def create_shm(self, name:str=None, max_size_mb:int=None, is_writer:bool=True):
        """
        Create a shared memory
        """
        if name is None:
            name = self.name
        if max_size_mb is None:
            max_size_mb = self.max_size_mb
        try:
            shm = SharedMemoryChannel(name=name, max_size_mb=max_size_mb, is_writer=is_writer)
        except Exception as e:
            raise ValueError(f"Failed to create shared memory: {e}")
        return shm


    def read_data_from_shm(self, shm:SharedMemoryChannel=None) -> dict:
        """
        Get the data from the shared memory
        """
        if shm is None: shm = self.shm
        if shm is not None:
            return shm.read()
        else:
            raise ValueError("Shared memory is not created")

    def write_data_to_shm(self, data: dict, shm:SharedMemoryChannel=None):
        """
        Write the data to the shared memory
        """
        if shm is None: shm = self.shm
        if shm is not None:
            shm.write(data)
        else:
            raise ValueError("Shared memory is not created")

    @abstractmethod
    def get_data(self) -> dict:
        """
        Update the data and return the data
        """
        pass

    def start(self):
        # create shared memory
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        self.is_running = True
        rate_limiter = RateLimiter()
        while self.is_running:
            data = self.get_data()
            if data is not None:
                self.write_data_to_shm(data)
            rate_limiter.sleep(self.fps)

    def close(self):
        """
        Close the shared memory
        """
        self.is_running = False
        if self.shm is not None:
            self.shm.destroy()
            self.shm = None

# Default obs2meta function
def default_obs2meta(synced_data: dict) -> MetaObs:
    """
    Default conversion from synced data to MetaObs.
    Extracts qpos from the first robot device and images from cameras.
    """
    qpos = None
    images = []

    def _append_image_candidate(img_value):
        nonlocal images
        if isinstance(img_value, np.ndarray):
            img = img_value
            if img.ndim == 3:  # HWC
                img = img.transpose(2, 0, 1)  # CHW
                img = img[np.newaxis, :]  # BCHW
            elif img.ndim == 4 and img.shape[-1] == 3:  # BHWC
                img = img.transpose(0, 3, 1, 2)  # BCHW
            else:
                return
            images.append(img)

    for dev_name, dev_data in synced_data.items():
        if not isinstance(dev_data, dict):
            continue
        
        # Extract qpos
        if 'qpos' in dev_data and qpos is None:
            qpos = np.array(dev_data['qpos'], dtype=np.float32)
        
        # Extract images from standard `image` field and flat camera-like fields.
        if 'image' in dev_data:
            _append_image_candidate(dev_data['image'])
        for key, value in dev_data.items():
            if key in ("qpos", "gpos", "action", "timestamp", "__timestamp__", "image"):
                continue
            _append_image_candidate(value)

    if qpos is None:
        qpos = np.zeros(6, dtype=np.float32)

    if images:
        image = np.concatenate(images, axis=0)
    else:
        image = np.zeros((1, 3, 480, 640), dtype=np.uint8)

    return MetaObs(state=qpos, image=image)
            
def is_robot_config(config:dict) -> bool:
    """
    Check if the config is a robot config
    """

    parts = config['type'].rsplit('.', 1)
    device_module_name = parts[0]
    device_class_name = parts[1]
    device_module = importlib.import_module(device_module_name)
    device_class = getattr(device_module, device_class_name)
    from deploy.robot.base import BaseRobot
    return issubclass(device_class, BaseRobot)

def is_camera_config(config:dict) -> bool:
    """
    Check if the config is a camera config
    """
    parts = config['type'].rsplit('.', 1)
    device_module_name = parts[0]
    device_class_name = parts[1]
    device_module = importlib.import_module(device_module_name)
    device_class = getattr(device_module, device_class_name)
    from deploy.sensor.camera.opencv_camera import OpenCVCamera
    return issubclass(device_class, OpenCVCamera)

def is_teleop_config(config:dict) -> bool:
    """
    Check if the config is a teleop config
    """
    parts = config['type'].rsplit('.', 1)
    device_module_name = parts[0]
    device_class_name = parts[1]
    device_module = importlib.import_module(device_module_name)
    device_class = getattr(device_module, device_class_name)
    from deploy.teleoperator.base import BaseTeleopDevice
    return issubclass(device_class, BaseTeleopDevice)