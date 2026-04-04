"""
Alicia-D Robot Module for ILStudio

Provides:
- AliciaD: Main robot class with qpos, delta_ee, and rel_ee control modes
- PiperStyleGlobalIKSolver: global IK used by AliciaD for EE modes
"""

from deploy.robot.alicia_d.piper_global_ik import PiperStyleGlobalIKSolver
from deploy.robot.alicia_d.robot import AliciaD

__all__ = ["AliciaD", "PiperStyleGlobalIKSolver"]
