# deploy/robots/base_robot.py

import abc
import importlib
from typing import Dict, Optional, Sequence, List, Any
import numpy as np
from benchmark.base import MetaAction, MetaObs
import time


class RateLimiter:
    """
    A class to manage the rate for a single thread.
    Each thread should have its own instance.
    """

    def __init__(self):
        self._last_sleep_time = time.perf_counter()

    def sleep(self, rate: float):
        """
        Sleeps for a duration that maintains the desired loop rate.

        Args:
            rate (float): The desired loop frequency in Hz.
        """
        if rate <= 0:
            return

        target_period = 1.0 / rate
        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_sleep_time
        sleep_duration = target_period - elapsed_time

        if sleep_duration > 0:
            time.sleep(sleep_duration)

        # Update the timestamp for the next iteration
        self._last_sleep_time = time.perf_counter()


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


class BaseRobot(AbstractRobotInterface):
    def meta2act(self, mact: MetaAction):
        """Convert the MetaAct to execusable actions for the robot"""
        return mact['action']

    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        return MetaObs(state=obs['qpos'], state_joint=obs['qpos'], image=np.stack([obs['image'][k] for k in obs['image']], axis=0).transpose(0, 3, 1, 2))

def make_robot(robot_cfg: Dict, args):
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

    robot = RobotCls(extra_args=args, **robot_config)
    # connect to robot
    retry_counts = 1
    MAX_RETRY = 5
    while not robot.connect():
        print(f"Retrying for {retry_counts} time...")
        retry_counts += 1
        if retry_counts>MAX_RETRY:
            exit(0)
        time.sleep(1)
    return robot