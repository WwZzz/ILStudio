from deploy.robot.base import BaseRobot
from .rosoperator import RosOperator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, List, Any
import numpy as np


class AgilexAloha(BaseRobot):
    def __init__(self, config, extra_args, *args, **kwargs):
        super().__init__()
        self.ros_operator = RosOperator(SimpleNamespace(**config['ros_operator']))
        self.init_pos = config.get('init_pos', {})

    def reset(self):
        try:
            left = self.init_pos.get('left', None)
            right = self.init_pos.get('right', None)
            if left is not None and right is not None:
                self.ros_operator.puppet_arm_publish_continuous(left0, right0)
            return True
        except Exception as e:
            print(f"Failed to reset robot to initial pos: \n Left: {left} \n Right: {right}")
            return False

    def connect(self):
        return self.reset()

    def get_observation(self) -> Dict[str, Any]:
        """
        """
        result = self.ros_operator.get_frame()
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result

        obs = {}
        # organize image
        obs['image'] = dict(primary=img_front, wrist_left=img_left, wrist_right=img_right)
        # organize depth
        if self.args.use_depth_image:
            obs['depth'] = dict(primary=img_front_depth, wrist_left=img_left_depth, wrist_right=img_right_depth)
        obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if self.args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        return obs

    def publish_action(self, action: np.ndarray):
        """
        Simulates publishing an action command to the robot and validates the action format.
        """
        left_action, right_action = action[:7], action[7:14]
        self.ros_operator.puppet_arm_publish(left_action, right_action)
        if self.args.use_robot_base:
            vel_action = action[14:16]
            self.ros_operator.robot_base_publish(vel_action)

    def is_running(self) -> bool:
        """
        Checks if the robot system is still running.
        A threading event is used to simulate system shutdown.
        """
        return not rospy.is_shutdown()

    def rate_sleep(self, hz: int):
        """
        Performs a sleep to control the frequency.
        """
        time.sleep(1.0 / hz)

    def shutdown(self):
        exit()

