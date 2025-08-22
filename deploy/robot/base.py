# deploy/robots/base_robot.py

import abc
from typing import Dict, Optional, Sequence, List, Any
import gymnasium as gym
import numpy as np
from benchmark.base import MetaAction, MetaObs


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
    def rate_sleep(self, hz: int):
        """Performs a sleep at a specified frequency."""
        pass
    
    @abc.abstractmethod
    def meta2act(self, mact):
        """Convert the MetaAct to execusable actions for the robot"""
        pass
    
    @abc.abstractmethod
    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        pass

class BaseRobot(AbstractRobotInterface):
    def meta2act(self, mact: MetaAction):
        """Convert the MetaAct to execusable actions for the robot"""
        try:
            return mact['action']
        except:
            print('ok')
            
    def obs2meta(self, obs):
        """Convert the observations from the robot to MetaObs"""
        return MetaObs(state=obs['qpos'], state_joint=obs['qpos'], image=np.stack([obs['image'][k] for k in obs['image']], axis=0).transpose(0, 3, 1, 2))