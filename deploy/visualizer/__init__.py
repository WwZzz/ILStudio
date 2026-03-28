"""
Visualizer module for displaying device data from shared memory.

Available visualizers:
- BaseVisualizer: Base class for custom visualizers
- CameraVisualizer: Display camera images in a grid
- MujocoCameraVisualizer: Display named camera keys from a robot SHM
"""

from deploy.visualizer.base import (
    BaseVisualizer,
    start_visualizer,
    get_visualizer_class,
    get_visualizer_type_string,
    start_all_visualizers,
    stop_all_visualizers,
)
from deploy.visualizer.camera_visualizer import (
    CameraVisualizer,
    start_camera_visualizer,
)
from deploy.visualizer.mujoco_camera_visualizer import (
    MujocoCameraVisualizer,
    start_mujoco_camera_visualizer,
)
from deploy.visualizer.mujoco_proxy_visualizer import (
    MujocoProxyVisualizer,
    start_mujoco_proxy_visualizer,
)

__all__ = [
    "BaseVisualizer",
    "CameraVisualizer",
    "MujocoCameraVisualizer",
    "MujocoProxyVisualizer",
    "start_visualizer",
    "start_camera_visualizer",
    "start_mujoco_camera_visualizer",
    "start_mujoco_proxy_visualizer",
    "get_visualizer_class",
    "get_visualizer_type_string",
    "start_all_visualizers",
    "stop_all_visualizers",
]
