#!/usr/bin/env python3
"""
Interactive web-based visualizer for episode data.
Provides frame-by-frame visualization of images and trajectory data with a slider control.
Adapted for IL-Studio repository HDF5 data format.
"""
import sys
import os

# Add repository root to path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_root)

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import h5py
import numpy as np
import base64
import io
from PIL import Image
import argparse
import glob
from typing import Dict, Any, Optional, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeVisualizer:
    def __init__(self, path: str):
        """Initialize the episode visualizer with HDF5 data or folder of episodes."""
        self.path = path
        self.data = {}
        self.episode_files = []
        self.current_episode = 0
        self.load_data()
        
        # Create Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self) -> None:
        """Load episode data from HDF5 file or folder of episodes."""
        logger.info(f"Loading episode data from: {self.path}")
        
        try:
            if os.path.isfile(self.path) and self.path.endswith('.hdf5'):
                # Single HDF5 file
                self.episode_files = [self.path]
                self.load_single_episode(0)
            elif os.path.isdir(self.path):
                # Folder containing multiple episodes
                self.episode_files = sorted(glob.glob(os.path.join(self.path, "*.hdf5")))
                if not self.episode_files:
                    raise ValueError(f"No HDF5 files found in directory: {self.path}")
                logger.info(f"Found {len(self.episode_files)} episode files")
                self.load_single_episode(0)
            else:
                raise ValueError(f"Path must be a valid HDF5 file or directory: {self.path}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_single_episode(self, episode_idx: int) -> None:
        """Load data from a single episode file."""
        if episode_idx >= len(self.episode_files):
            logger.warning(f"Episode index {episode_idx} out of range")
            return
            
        file_path = self.episode_files[episode_idx]
        logger.info(f"Loading episode {episode_idx}: {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Recursively load all datasets
                self.data = {}
                self._recursive_load_data(f, self.data, "")
            
            # Extract common data patterns for IL-Studio format
            self._extract_common_data_patterns()
            
            # Determine the maximum number of frames
            if self.data:
                self.max_frames = self._get_max_frames()
                logger.info(f"Maximum frames: {self.max_frames}")
            else:
                self.max_frames = 0
                
        except Exception as e:
            logger.error(f"Error loading episode {episode_idx}: {e}")
            raise
    
    def _recursive_load_data(self, group, data_dict, path_prefix=""):
        """Recursively load data from HDF5 group."""
        for key in group.keys():
            item = group[key]
            current_path = f"{path_prefix}/{key}" if path_prefix else key
            
            if isinstance(item, h5py.Group):
                # Recurse into subgroups
                self._recursive_load_data(item, data_dict, current_path)
            elif isinstance(item, h5py.Dataset):
                # Load dataset
                try:
                    data_dict[current_path] = item[...]
                    logger.info(f"Loaded {current_path}: {data_dict[current_path].shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {current_path}: {e}")
    
    def _extract_common_data_patterns(self):
        """Extract and organize common data patterns from IL-Studio datasets."""
        # Find observations data
        self.observations = {}
        self.actions = {}
        self.images = {}
        
        for key, data in self.data.items():
            # Extract image data
            if 'image' in key.lower() or 'camera' in key.lower():
                if len(data.shape) == 4:  # T x H x W x C
                    camera_name = self._extract_camera_name(key)
                    self.images[camera_name] = data
                elif len(data.shape) == 3:  # H x W x C (single frame)
                    camera_name = self._extract_camera_name(key)
                    self.images[camera_name] = data[np.newaxis, ...]  # Add time dimension
            
            # Extract observation data (joint positions, etc.)
            elif 'observations' in key:
                obs_name = key.split('/')[-1]
                self.observations[obs_name] = data
            
            # Extract action data
            elif 'action' in key.lower():
                action_name = key.split('/')[-1] 
                self.actions[action_name] = data
        
        logger.info(f"Found {len(self.images)} camera streams")
        logger.info(f"Found {len(self.observations)} observation streams")
        logger.info(f"Found {len(self.actions)} action streams")
    
    def _extract_camera_name(self, key):
        """Extract camera name from key path."""
        # Handle different naming conventions
        if 'front' in key.lower():
            return 'front_camera'
        elif 'wrist' in key.lower():
            return 'wrist_camera'
        elif 'side' in key.lower():
            return 'side_camera'
        elif 'top' in key.lower():
            return 'top_camera'
        else:
            # Extract last meaningful part of the path
            parts = key.split('/')
            for part in reversed(parts):
                if part and 'image' not in part.lower():
                    return part
            return 'camera'
    
    def _get_max_frames(self):
        """Get maximum number of frames across all temporal data."""
        max_frames = 0
        for data in self.data.values():
            if len(data.shape) > 0:
                max_frames = max(max_frames, data.shape[0])
        return max_frames
    
    def image_to_base64(self, img_array: np.ndarray) -> str:
        """Convert numpy image array to base64 string for display."""
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        encoded = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{encoded}"
    
    def get_robot_data(self) -> Dict[str, np.ndarray]:
        """Extract robot joint/end-effector data."""
        robot_data = {}
        
        # Look for common robot data patterns
        for key, data in self.data.items():
            # Joint positions
            if any(joint in key.lower() for joint in ['qpos', 'joint', 'shoulder', 'elbow', 'wrist', 'gripper']):
                if len(data.shape) >= 1:
                    robot_data[key] = data
            
            # End-effector positions  
            elif any(ef in key.lower() for ef in ['eef', 'end_effector', 'tcp', 'pose']):
                if len(data.shape) >= 1:
                    robot_data[key] = data
                    
            # Action data
            elif 'action' in key.lower():
                if len(data.shape) >= 1:
                    robot_data[key] = data
        
        return robot_data
    
    def get_observation_data(self) -> Dict[str, np.ndarray]:
        """Extract observation data (states, sensors, etc.)."""
        obs_data = {}
        
        for key, data in self.data.items():
            # Look for observation patterns
            if 'observations' in key or 'state' in key.lower():
                if len(data.shape) >= 1:
                    obs_data[key] = data
                    
        return obs_data
    
    def get_end_effector_positions(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract end-effector position data specifically."""
        ee_data = {}
        
        for key, data in self.data.items():
            # Look for position data that could be end-effector
            if len(data.shape) == 2 and data.shape[1] >= 3:  # At least x,y,z
                if any(pos_key in key.lower() for pos_key in ['pos', 'position', 'eef', 'tcp']):
                    ee_data[key] = data[:, :3]  # Extract x,y,z
                elif 'action' in key.lower() and data.shape[1] >= 6:  # Could be action with position+rotation
                    ee_data[f"{key}_pos"] = data[:, :3]  # Position part
                    if data.shape[1] >= 6:
                        ee_data[f"{key}_rot"] = data[:, 3:6]  # Rotation part
        
        return ee_data if ee_data else None
    
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [qx, qy, qz, qw] to rotation matrix."""
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ])
        return R
    
    def create_coordinate_frame(self, position: np.ndarray, quaternion: np.ndarray, scale: float = 0.1) -> List[go.Scatter3d]:
        """Create coordinate frame visualization for a given position and orientation."""
        # Convert quaternion to rotation matrix
        R = self.quaternion_to_rotation_matrix(quaternion)
        
        # Define coordinate frame axes (X, Y, Z)
        axes = np.array([
            [scale, 0, 0],  # X axis (red)
            [0, scale, 0],  # Y axis (green)  
            [0, 0, scale]   # Z axis (blue)
        ])
        
        # Rotate axes according to orientation
        rotated_axes = (R @ axes.T).T
        
        # Create traces for each axis
        traces = []
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i, (axis, color, label) in enumerate(zip(rotated_axes, colors, labels)):
            # Start and end points for the axis
            start = position
            end = position + axis
            
            # Create line trace
            traces.append(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]], 
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color=color, width=8),
                name=f'{label} axis',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add arrow head (small sphere at the end)
            traces.append(go.Scatter3d(
                x=[end[0]],
                y=[end[1]], 
                z=[end[2]],
                mode='markers',
                marker=dict(size=6, color=color, symbol='diamond'),
                name=f'{label} tip',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        return traces
    
    def setup_layout(self) -> None:
        """Setup the Dash app layout."""
        # Use extracted data patterns
        image_datasets = list(self.images.keys())
        trajectory_datasets = list(self.observations.keys()) + list(self.actions.keys())
        
        # Get robot and observation data for layout decisions
        robot_data = self.get_robot_data()
        obs_data = self.get_observation_data()
        ee_data = self.get_end_effector_positions()
        
        self.app.layout = html.Div([
            html.H1("Episode Data Visualizer", style={'textAlign': 'center'}),
            
            # Episode selector (only show if multiple episodes)
            html.Div([
                html.Label("Episode:"),
                dcc.Dropdown(
                    id='episode-dropdown',
                    options=[
                        {'label': f'Episode {i}: {os.path.basename(file)}', 'value': i}
                        for i, file in enumerate(self.episode_files)
                    ],
                    value=0,
                    style={'width': '100%'}
                )
            ], style={'margin': '20px', 'display': 'block' if len(self.episode_files) > 1 else 'none'}),
            
            # Frame slider
            html.Div([
                html.Label("Frame:"),
                dcc.Slider(
                    id='frame-slider',
                    min=0,
                    max=self.max_frames - 1,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, self.max_frames, max(1, self.max_frames // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px'}),
            
            # Images section
            html.Div([
                html.H3("Camera Images"),
                html.Div(
                    id='images-container',
                    children=[
                        html.Div([
                            html.H4(camera_name),
                            html.Img(id=f'image-{camera_name}', style={'width': '100%', 'max-width': '640px'})
                        ], style={'display': 'inline-block', 'width': '48%', 'margin': '1%'})
                        for camera_name in image_datasets
                    ]
                )
            ]) if image_datasets else html.Div(),
            
            # 3D Robot Trajectory section
            html.Div([
                html.H3("Robot Data Visualization"),
                html.Div([
                    dcc.Graph(id='3d-trajectory-plot', style={'height': '600px'}),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='robot-positions-plot')
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(id='robot-actions-plot')
                        ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '2%'})
                    ])
                ])
            ]) if robot_data or ee_data else html.Div(),
            
            # Observations section
            html.Div([
                html.H3("Observation Data"),
                html.Div([
                    dcc.Graph(id='observations-plot', style={'height': '400px'}),
                ])
            ]) if obs_data else html.Div(),
            
            # Trajectory plots section
            html.Div([
                html.H3("All Trajectory Data"),
                html.Div([
                    dcc.Graph(id='trajectory-plot'),
                    dcc.Graph(id='trajectory-values')
                ])
            ]),
            
            # Data info
            html.Div([
                html.H3("Data Information"),
                html.Div(id='data-info')
            ], style={'margin': '20px'})
        ])
    
    def setup_callbacks(self) -> None:
        """Setup Dash callbacks for interactivity."""
        
        # Update episode data when episode dropdown changes
        @callback(
            [Output('frame-slider', 'max'),
             Output('frame-slider', 'marks'),
             Output('frame-slider', 'value')],
            [Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_episode(episode_idx):
            if episode_idx is not None and episode_idx != self.current_episode:
                self.current_episode = episode_idx
                self.load_single_episode(episode_idx)
            
            max_frames = self.max_frames - 1 if self.max_frames > 0 else 0
            marks = {i: str(i) for i in range(0, self.max_frames, max(1, self.max_frames // 10))}
            
            return max_frames, marks, 0
        
        # Update images based on frame slider
        for camera_name in self.images.keys():
            @callback(
                Output(f'image-{camera_name}', 'src'),
                [Input('frame-slider', 'value'),
                 Input('episode-dropdown', 'value')],
                prevent_initial_call=False
            )
            def update_image(frame_idx, episode_idx, camera_name=camera_name):
                if camera_name in self.images and frame_idx < self.images[camera_name].shape[0]:
                    img = self.images[camera_name][frame_idx]
                    return self.image_to_base64(img)
                return ""
        
        # Update 3D trajectory plot
        @callback(
            [Output('3d-trajectory-plot', 'figure'),
             Output('robot-positions-plot', 'figure'),
             Output('robot-actions-plot', 'figure')],
            [Input('frame-slider', 'value'),
             Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_3d_trajectory(frame_idx, episode_idx):
            robot_data = self.get_robot_data()
            ee_data = self.get_end_effector_positions()
            
            if not robot_data and not ee_data:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No robot data available")
                return empty_fig, empty_fig, empty_fig
            
            # Create 3D trajectory plot
            fig_3d = go.Figure()
            
            # Plot end-effector trajectories if available
            if ee_data:
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, (name, data) in enumerate(ee_data.items()):
                    if len(data.shape) == 2 and data.shape[1] >= 3 and frame_idx < data.shape[0]:
                        color = colors[i % len(colors)]
                        trajectory = data[:frame_idx+1, :3]  # x,y,z
                        
                        # Plot trajectory
                        fig_3d.add_trace(go.Scatter3d(
                            x=trajectory[:, 0],
                            y=trajectory[:, 1],
                            z=trajectory[:, 2],
                            mode='lines+markers',
                            name=f'{name} Trajectory',
                            line=dict(color=color, width=4),
                            marker=dict(size=3)
                        ))
                        
                        # Current position marker
                        fig_3d.add_trace(go.Scatter3d(
                            x=[data[frame_idx, 0]],
                            y=[data[frame_idx, 1]],
                            z=[data[frame_idx, 2]],
                            mode='markers',
                            name=f'{name} Current',
                            marker=dict(size=10, color=color, symbol='diamond')
                        ))
            
            fig_3d.update_layout(
                title=f"3D Trajectories (Frame {frame_idx})",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode='cube'
                ),
                height=600
            )
            
            # Create robot positions plot
            fig_positions = go.Figure()
            frames = list(range(frame_idx + 1))
            
            if ee_data:
                colors = ['red', 'green', 'blue']
                for name, data in ee_data.items():
                    if len(data.shape) == 2 and data.shape[1] >= 3 and frame_idx < data.shape[0]:
                        positions = data[:frame_idx+1, :3]
                        for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], colors)):
                            fig_positions.add_trace(go.Scatter(
                                x=frames, y=positions[:, i], 
                                name=f'{name} {axis}', 
                                line=dict(color=color, dash='solid' if i == 0 else 'dash' if i == 1 else 'dot')
                            ))
                            
                            # Current frame marker
                            fig_positions.add_trace(go.Scatter(
                                x=[frame_idx], y=[data[frame_idx, i]],
                                mode='markers', marker=dict(size=8, color=color),
                                name=f'Current {name} {axis}', showlegend=False
                            ))
            
            fig_positions.update_layout(
                title="Position Data Over Time",
                xaxis_title="Frame",
                yaxis_title="Position"
            )
            
            # Create robot actions plot
            fig_actions = go.Figure()
            
            action_data = [data for key, data in robot_data.items() if 'action' in key.lower()]
            if action_data:
                data = action_data[0]  # Use first action dataset
                if len(data.shape) == 2 and frame_idx < data.shape[0]:
                    # Plot first few dimensions of action data
                    for i in range(min(6, data.shape[1])):
                        values = data[:frame_idx+1, i]
                        fig_actions.add_trace(go.Scatter(
                            x=frames, y=values,
                            name=f'Action Dim {i}',
                            line=dict(width=2)
                        ))
                        
                        # Current frame marker
                        fig_actions.add_trace(go.Scatter(
                            x=[frame_idx], y=[data[frame_idx, i]],
                            mode='markers', marker=dict(size=8),
                            name=f'Current Action {i}', showlegend=False
                        ))
            
            fig_actions.update_layout(
                title="Action Data Over Time",
                xaxis_title="Frame",
                yaxis_title="Action Value"
            )
            
            return fig_3d, fig_positions, fig_actions
        
        # Update observations plot
        @callback(
            Output('observations-plot', 'figure'),
            [Input('frame-slider', 'value'),
             Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_observations(frame_idx, episode_idx):
            obs_data = self.get_observation_data()
            
            if not obs_data:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No observation data available")
                return empty_fig
            
            fig = go.Figure()
            frames = list(range(frame_idx + 1))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, (obs_name, data) in enumerate(obs_data.items()):
                if len(data.shape) >= 1 and frame_idx < data.shape[0]:
                    color = colors[i % len(colors)]
                    
                    # For multi-dimensional data, plot first few dimensions
                    if len(data.shape) == 2:
                        for dim in range(min(3, data.shape[1])):  # Plot first 3 dimensions
                            values = data[:frame_idx+1, dim]
                            fig.add_trace(go.Scatter(
                                x=frames, y=values,
                                name=f'{obs_name}_dim{dim}',
                                line=dict(color=color, dash='solid' if dim == 0 else 'dash' if dim == 1 else 'dot')
                            ))
                            
                            # Current frame marker
                            fig.add_trace(go.Scatter(
                                x=[frame_idx], y=[data[frame_idx, dim]],
                                mode='markers', marker=dict(size=8, color=color),
                                name=f'Current {obs_name}_dim{dim}', showlegend=False
                            ))
                    elif len(data.shape) == 1:
                        # Single dimension data
                        values = data[:frame_idx+1]
                        fig.add_trace(go.Scatter(
                            x=frames, y=values,
                            name=obs_name,
                            line=dict(color=color, width=2)
                        ))
                        
                        # Current frame marker
                        fig.add_trace(go.Scatter(
                            x=[frame_idx], y=[data[frame_idx]],
                            mode='markers', marker=dict(size=8, color=color),
                            name=f'Current {obs_name}', showlegend=False
                        ))
            
            fig.update_layout(
                title="Observation Data Over Time",
                xaxis_title="Frame",
                yaxis_title="Value"
            )
            
            return fig
        
        # Update trajectory plot
        @callback(
            [Output('trajectory-plot', 'figure'),
             Output('trajectory-values', 'figure'),
             Output('data-info', 'children')],
            [Input('frame-slider', 'value'),
             Input('episode-dropdown', 'value')],
            prevent_initial_call=False
        )
        def update_trajectory(frame_idx, episode_idx):
            # Get all data for trajectory plotting
            all_data = {**self.observations, **self.actions}
            robot_data = self.get_robot_data()
            
            # Create trajectory overview plot
            fig_traj = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            color_idx = 0
            
            for dataset_name, data in all_data.items():
                if len(data.shape) == 2 and frame_idx < data.shape[0]:
                    # Plot first few dimensions
                    for i in range(min(3, data.shape[1])):
                        color = colors[color_idx % len(colors)]
                        fig_traj.add_trace(go.Scatter(
                            x=list(range(frame_idx + 1)),
                            y=data[:frame_idx + 1, i],
                            mode='lines',
                            name=f'{dataset_name}_dim_{i}',
                            line=dict(width=2, color=color)
                        ))
                        
                        # Add current frame marker
                        fig_traj.add_trace(go.Scatter(
                            x=[frame_idx],
                            y=[data[frame_idx, i]],
                            mode='markers',
                            marker=dict(size=8, color=color),
                            name=f'Current_{dataset_name}_dim_{i}',
                            showlegend=False
                        ))
                        color_idx += 1
                elif len(data.shape) == 1 and frame_idx < data.shape[0]:
                    color = colors[color_idx % len(colors)]
                    fig_traj.add_trace(go.Scatter(
                        x=list(range(frame_idx + 1)),
                        y=data[:frame_idx + 1],
                        mode='lines',
                        name=dataset_name,
                        line=dict(width=2, color=color)
                    ))
                    
                    # Add current frame marker
                    fig_traj.add_trace(go.Scatter(
                        x=[frame_idx],
                        y=[data[frame_idx]],
                        mode='markers',
                        marker=dict(size=8, color=color),
                        name=f'Current_{dataset_name}',
                        showlegend=False
                    ))
                    color_idx += 1
            
            fig_traj.update_layout(
                title="All Data Over Time",
                xaxis_title="Frame",
                yaxis_title="Value",
                height=400
            )
            
            # Create current frame values plot
            fig_values = go.Figure()
            
            # Collect current values for bar chart
            bar_labels = []
            bar_values = []
            bar_colors = []
            
            for dataset_name, data in all_data.items():
                if frame_idx < data.shape[0]:
                    if len(data.shape) == 2:
                        for i in range(min(6, data.shape[1])):  # Show first 6 dimensions
                            bar_labels.append(f'{dataset_name}_dim_{i}')
                            bar_values.append(data[frame_idx, i])
                            bar_colors.append(colors[len(bar_labels) % len(colors)])
                    elif len(data.shape) == 1:
                        bar_labels.append(dataset_name)
                        bar_values.append(data[frame_idx])
                        bar_colors.append(colors[len(bar_labels) % len(colors)])
            
            if bar_labels:
                fig_values.add_trace(go.Bar(
                    x=bar_labels,
                    y=bar_values,
                    marker_color=bar_colors,
                    name="Current Values"
                ))
            
            fig_values.update_layout(
                title=f"Current Frame Values (Frame {frame_idx})",
                xaxis_title="Data Field",
                yaxis_title="Value",
                height=400,
                xaxis_tickangle=-45
            )
            
            # Create data info
            info_text = []
            
            # Episode information
            info_text.append(html.H4("Episode Information"))
            info_text.append(html.P([
                html.Strong("Current Episode: "), f"{episode_idx + 1} of {len(self.episode_files)}",
                html.Br(),
                html.Strong("File: "), os.path.basename(self.episode_files[episode_idx]),
                html.Br(),
                html.Strong("Current Frame: "), f"{frame_idx + 1} of {self.max_frames}",
                html.Br(),
                html.Strong("Total Data Fields: "), f"{len(self.data)}"
            ]))
            
            # Camera information
            if self.images:
                info_text.append(html.H4("Camera Data"))
                for camera_name, image_data in self.images.items():
                    if frame_idx < image_data.shape[0]:
                        info_text.append(html.P([
                            html.Strong(f"{camera_name}: "),
                            f"Shape: {image_data.shape}, Current frame: {image_data.shape[1]}x{image_data.shape[2]} pixels"
                        ]))
            
            # Robot/Action data information
            robot_data = self.get_robot_data()
            if robot_data:
                info_text.append(html.H4("Robot Data"))
                for name, data in robot_data.items():
                    if len(data.shape) >= 1 and frame_idx < data.shape[0]:
                        if len(data.shape) == 2:
                            current_values = data[frame_idx][:min(6, data.shape[1])]  # Show first 6 values
                            values_str = ", ".join([f"{v:.4f}" for v in current_values])
                            if data.shape[1] > 6:
                                values_str += "..."
                        else:
                            values_str = f"{data[frame_idx]:.4f}"
                        
                        info_text.append(html.P([
                            html.Strong(f"{name}: "),
                            f"Shape: {data.shape}, Current: [{values_str}]"
                        ]))
            
            # Observation data information
            obs_data = self.get_observation_data()
            if obs_data:
                info_text.append(html.H4("Observation Data"))
                for name, data in obs_data.items():
                    if len(data.shape) >= 1 and frame_idx < data.shape[0]:
                        if len(data.shape) == 2:
                            current_values = data[frame_idx][:min(6, data.shape[1])]  # Show first 6 values
                            values_str = ", ".join([f"{v:.4f}" for v in current_values])
                            if data.shape[1] > 6:
                                values_str += "..."
                        else:
                            values_str = f"{data[frame_idx]:.4f}"
                        
                        info_text.append(html.P([
                            html.Strong(f"{name}: "),
                            f"Shape: {data.shape}, Current: [{values_str}]"
                        ]))
            
            # End-effector positions if available
            ee_data = self.get_end_effector_positions()
            if ee_data:
                info_text.append(html.H4("End-Effector Positions"))
                for name, data in ee_data.items():
                    if frame_idx < data.shape[0] and data.shape[1] >= 3:
                        pos = data[frame_idx, :3]
                        info_text.append(html.P([
                            html.Strong(f"{name}: "),
                            f"X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}"
                        ]))
            
            # Raw data summary
            info_text.append(html.H4("Raw Data Summary"))
            for key, data in self.data.items():
                if len(data.shape) > 0:
                    info_text.append(html.P([
                        html.Strong(f"{key}: "),
                        f"Shape: {data.shape}, Type: {data.dtype}"
                    ]))
            
            return fig_traj, fig_values, info_text
    
    def run(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True) -> None:
        """Run the Dash application."""
        logger.info(f"Starting visualizer at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Interactive episode data visualizer")
    parser.add_argument("path", help="Path to HDF5 episode file or folder containing multiple episodes")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run server on")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    
    args = parser.parse_args()
    
    try:
        visualizer = EpisodeVisualizer(args.path)
        visualizer.run(host=args.host, port=args.port, debug=not args.no_debug)
    except Exception as e:
        logger.error(f"Failed to start visualizer: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 