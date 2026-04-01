"""
Visualizer module for displaying device data from shared memory.

Available visualizers:
- BaseVisualizer: Base class for custom visualizers
- CameraVisualizer: Display camera images in a grid
- LowDimVisualizer: Time-series plots for low-dimensional SHM vectors (see lowdim_visualizer.py)
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

__all__ = [
    "BaseVisualizer",
    "CameraVisualizer",
    "LowDimVisualizer",
    "start_visualizer",
    "start_camera_visualizer",
    "start_lowdim_visualizer",
    "get_visualizer_class",
    "get_visualizer_type_string",
    "start_all_visualizers",
    "stop_all_visualizers",
]
