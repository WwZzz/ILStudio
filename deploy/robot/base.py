# deploy/robots/base_robot.py

import abc
import importlib
from typing import Dict, Optional, Sequence, List, Any
import numpy as np
from benchmark.base import MetaAction, MetaObs
from deploy.utils import RateLimiter
from deploy.base import BaseDevice
import time
from loguru import logger

class AbstractRobotInterface(abc.ABC):
    """Defines the abstract base class for a robot interface."""

    @abc.abstractmethod
    def connect(self):
        """Connects to the robot SDK or system."""
        pass

    @abc.abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """
        Retrieves a synchronized, complete multimodal observation.
        This method is designed to be blocking to ensure data integrity.
        """
        pass

    @abc.abstractmethod
    def publish_action(self, action: np.ndarray):
        """
        Publishes an action command to the robot.
        This method is designed to be non-blocking to ensure the smoothness of the control loop.
        """
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        """Checks if the robot system is still running."""
        pass

    @abc.abstractmethod
    def meta2act(self, mact):
        """Convert the MetaAct to execusable actions for the robot"""
        pass

    @abc.abstractmethod
    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        pass

    @abc.abstractmethod
    def shutdown(self):
        """Disconnect the robot and shutdown"""
        pass
    
    @abc.abstractmethod
    def get_action_dim(self):
        """Return the shape of the action space"""
        pass

class BaseRobot(BaseDevice, AbstractRobotInterface):
    def __init__(self, name:str, max_size_mb:int=64, fps:float=1000, control_shm_name:Optional[str]=None, **kwargs):
        BaseDevice.__init__(self, name, max_size_mb, fps)
        AbstractRobotInterface.__init__(self)
        self.control_shm_name = control_shm_name
        # Note: control_shm connection is deferred to start() to allow teleop shm to be created first
        self.control_shm = None

    def read_action(self) -> dict:
        if self.control_shm is not None:
            return self.control_shm.read(skip_unchanged=True)
        else:
            return None

    def process_action(self, x):
        """Process the action (may be slow, e.g., IK computation)"""
        return x

    def meta2act(self, mact: MetaAction):
        """Convert the MetaAct to execusable actions for the robot"""
        return mact['action']

    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        return MetaObs(state=obs['qpos'], state_joint=obs['qpos'], image=np.stack([obs['image'][k] for k in obs['image']], axis=0).transpose(0, 3, 1, 2))


    def get_data(self) -> dict:
        """Get the data from the robot"""
        return self.get_observation()

    def write_data(self, data: dict):
        """Write the robot data to the shared memory"""
        assert self.shm is not None, "Shared memory is not created"
        try:
            self.shm.write(data)
        except Exception as e:
            logger.error(f"Failed to write data to shared memory: {e}")

    def start(self):
        """
        Start the robot.
        
        IMPORTANT: process_action() MUST be fast (< 5ms) to maintain observation rate.
        IK computation should use fast_mode=True with minimal iterations.
        """
        # create shared memory for robot observations
        self.shm = self.create_shm(name=self.name, max_size_mb=self.max_size_mb, is_writer=True)
        
        # Connect to control shm (teleop) with retry - it may not be ready yet
        if self.control_shm_name is not None and self.control_shm is None:
            max_retries = 10
            for i in range(max_retries):
                try:
                    self.control_shm = self.connect_to_existing_shm(self.control_shm_name)
                    logger.info(f"Connected to control SHM: {self.control_shm_name}")
                    break
                except ValueError as e:
                    if i < max_retries - 1:
                        logger.warning(f"Waiting for control SHM '{self.control_shm_name}'... ({i+1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        logger.error(f"Failed to connect to control SHM: {e}")
        
        self.is_running = True
        rate_limiter = RateLimiter()
        
        while self.is_running:
            # 1. Get and publish observation
            data = self.get_data()
            if data is not None:
                self.write_data(data)
            
            # 2. Process and execute action (MUST be fast!)
            action = self.read_action()
            if action is not None:
                action = self.process_action(action)
                action_array = action.get('action', None)
                if action_array is not None:
                    self.publish_action(action_array)
            
            rate_limiter.sleep(self.fps)


def make_robot(robot_cfg: Dict, args, max_connect_retry: int = 5):
    """
    Factory function to create a robot instance from a config dictionary.

    Args:
        robot_cfg (Dict): A dictionary loaded from the robot's YAML config file.
        args: Command line arguments (not used by robots).
        max_connect_retry (int): Maximum number of connection retries. Defaults to 5.
    """
    full_path = robot_cfg['type']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    RobotCls = getattr(module, class_name)
    print(f"Creating robot: {full_path}")

    # Create a copy of robot_cfg without the 'target' key for passing as kwargs
    if 'args' in robot_cfg:
        robot_kwargs = {k: v for k, v in robot_cfg['args'].items()}
    else:
        robot_kwargs = {}
    robot = RobotCls(**robot_kwargs)
    # connect to robot
    retry_counts = 1
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts > max_connect_retry:
            exit(0)
        time.sleep(1)
    return robot