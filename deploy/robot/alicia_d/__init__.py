"""
Alicia-D Robot Module for ILStudio

Provides:
- AliciaD: Main robot class with qpos and delta_ee control modes
- Uses fast Pinocchio-based IK solver (~0.3ms per solve)
"""

from deploy.robot.alicia_d.robot import AliciaD, RealtimeIKSolver

__all__ = ["AliciaD", "RealtimeIKSolver"]
