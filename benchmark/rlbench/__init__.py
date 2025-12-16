import os
import sys
from pathlib import Path

# Setup CoppeliaSim environment variables
# This ensures RLBench/PyRep can find CoppeliaSim libraries
if 'COPPELIASIM_ROOT' not in os.environ:
    # Default path, adjust if your CoppeliaSim is installed elsewhere
    default_coppeliasim_path = os.path.expanduser('~/CoppeliaSim')
    if os.path.exists(default_coppeliasim_path):
        os.environ['COPPELIASIM_ROOT'] = default_coppeliasim_path
    else:
        print(f"Warning: CoppeliaSim not found at {default_coppeliasim_path}")
        print("Please set COPPELIASIM_ROOT environment variable or install CoppeliaSim")

if 'COPPELIASIM_ROOT' in os.environ:
    coppeliasim_root = os.environ['COPPELIASIM_ROOT']
    # Add to LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' in os.environ:
        if coppeliasim_root not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{coppeliasim_root}"
    else:
        os.environ['LD_LIBRARY_PATH'] = coppeliasim_root
    
    # Set Qt plugin path
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = coppeliasim_root

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from ..base import MetaEnv, MetaObs, MetaAction
import importlib


def create_env(config):
    """
    Create RLBench environment instance.
    
    IMPORTANT: RLBench does NOT support parallel execution (SubprocVectorEnv)
    because CoppeliaSim cannot safely run in multiple processes.
    
    When using eval_sim.py, always use --batch_size 0 to force sequential mode:
        python eval_sim.py -e rlbench_reach -m <model> --batch_size 0
    
    Args:
        config: Configuration object containing task parameters
    
    Returns:
        RLBenchEnv: RLBench environment instance
    """
    import multiprocessing
    
    # Warn if this appears to be a subprocess (not main process)
    if multiprocessing.current_process().name != 'MainProcess':
        import warnings
        warnings.warn(
            "\n" + "="*70 + "\n"
            "WARNING: RLBench is being created in a subprocess!\n"
            "RLBench does NOT support parallel execution (SubprocVectorEnv).\n"
            "This will likely fail. Please use --batch_size 0 for sequential mode:\n"
            "    python eval_sim.py -e rlbench_reach -m <model> --batch_size 0\n"
            + "="*70,
            RuntimeWarning,
            stacklevel=2
        )
    
    return RLBenchEnv(config)


class RLBenchEnv(MetaEnv):
    """
    RLBench environment wrapper for ILStudio.
    
    Supports various RLBench tasks with configurable observation and action modes.
    """
    
    def __init__(self, config, *args):
        """
        Initialize RLBench environment.
        
        Args:
            config: Configuration object with the following attributes:
                - task: Task name (e.g., 'ReachTarget', 'PickAndLift')
                - ctrl_space: 'joint' or 'ee' (default: 'joint')
                - ctrl_type: 'abs' or 'delta' (default: 'abs')
                - camera_names: List of camera names (default: ['front', 'wrist'])
                - image_size: Image resolution (default: [128, 128])
                - headless: Run without GUI (default: True)
        """
        self.config = config
        self.task_name = config.task
        self.ctrl_space = getattr(config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(config, 'ctrl_type', 'abs')
        self.camera_names = getattr(config, 'camera_names', ['front', 'wrist'])
        self.image_size = getattr(config, 'image_size', [128, 128])
        self.headless = getattr(config, 'headless', True)
        
        # Create RLBench environment
        env = self._create_rlbench_env()
        
        # Store task and descriptions
        self.rlbench_env = env
        self.task = None
        self.descriptions = []
        self.raw_lang = ""
        
        # Don't call super().__init__ with env since we handle task separately
        self.env = None
        self.prev_obs = None
        
    def _create_rlbench_env(self):
        """Create RLBench environment with appropriate action mode."""
        # Configure observation
        obs_config = ObservationConfig()
        obs_config.set_all(True)  # Enable all observations
        
        # Configure action mode based on ctrl_space
        if self.ctrl_space == 'joint':
            arm_action_mode = JointVelocity()
        elif self.ctrl_space == 'ee':
            arm_action_mode = EndEffectorPoseViaPlanning()
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")
        
        gripper_action_mode = Discrete()
        action_mode = MoveArmThenGripper(
            arm_action_mode=arm_action_mode,
            gripper_action_mode=gripper_action_mode
        )
        
        # Create environment
        env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=self.headless,
            shaped_rewards=False
        )
        env.launch()
        
        return env
    
    def _get_task_class(self, task_name):
        """Dynamically load task class by name."""
        try:
            # Import from rlbench.tasks
            task_module = importlib.import_module('rlbench.tasks')
            task_class = getattr(task_module, task_name)
            return task_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Task '{task_name}' not found in rlbench.tasks: {e}")
    
    def obs2meta(self, obs):
        """
        Convert RLBench observation to MetaObs format.
        
        RLBench observation structure:
        - joint_positions: (7,) joint angles
        - joint_velocities: (7,) joint velocities
        - gripper_open: (1,) gripper state
        - gripper_pose: (7,) [x, y, z, qx, qy, qz, qw]
        - gripper_joint_positions: (2,) gripper joint positions
        - task_low_dim_state: task-specific state
        - {camera}_rgb: (H, W, 3) RGB images
        - {camera}_depth: (H, W) depth images
        """
        # Extract robot state based on ctrl_space
        if self.ctrl_space == 'joint':
            # Joint positions + gripper state
            joint_positions = obs.joint_positions  # (7,)
            gripper_state = obs.gripper_open  # scalar - 1.0 for open, 0.0 for closed
            # Ensure gripper_state is 1D array
            gripper_state = np.array([gripper_state]) if np.isscalar(gripper_state) else gripper_state
            state = np.concatenate([joint_positions, gripper_state], axis=0).astype(np.float32)
        elif self.ctrl_space == 'ee':
            # End-effector pose + gripper state
            gripper_pose = obs.gripper_pose  # (7,) [x, y, z, qx, qy, qz, qw]
            gripper_state = obs.gripper_open  # scalar
            # Ensure gripper_state is 1D array
            gripper_state = np.array([gripper_state]) if np.isscalar(gripper_state) else gripper_state
            state = np.concatenate([gripper_pose, gripper_state], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")
        
        # Extract images from specified cameras
        images = []
        for cam_name in self.camera_names:
            # RLBench camera names: front, left_shoulder, right_shoulder, wrist, overhead
            cam_attr = f"{cam_name}_rgb"
            if hasattr(obs, cam_attr):
                img = getattr(obs, cam_attr)  # (H, W, 3) in RGB format
                images.append(img)
        
        # Stack and transpose to (N, C, H, W) format
        if len(images) > 0:
            image = np.stack(images)  # (N, H, W, 3)
            image = image.transpose(0, 3, 1, 2)  # (N, 3, H, W)
        else:
            image = None
        
        return MetaObs(
            state=state,
            state_joint=obs.joint_positions if hasattr(obs, 'joint_positions') else None,
            state_ee=obs.gripper_pose if hasattr(obs, 'gripper_pose') else None,
            image=image,
            raw_lang=self.raw_lang
        )
    
    def meta2act(self, maction: MetaAction):
        """
        Convert MetaAction to RLBench action format.
        
        RLBench action format depends on action mode:
        - JointVelocity: (7,) joint velocities + (1,) gripper action
        - EndEffectorPoseViaPlanning: (7,) [x, y, z, qx, qy, qz, qw] + (1,) gripper
        
        Gripper action: 0 = close, 1 = open (discrete)
        """
        action = maction['action']  # (action_dim,)
        
        # Handle dimension mismatch: add gripper if missing
        if len(action) == 7:
            # Add default gripper action (open = 1.0)
            action = np.concatenate([action, [1.0]])
        
        # RLBench expects gripper to be discrete: 0 or 1
        # Convert continuous gripper value to discrete
        rlbench_action = np.array(action, copy=True).astype(np.float32)
        
        # For end-effector control, normalize quaternion
        if self.ctrl_space == 'ee' and len(rlbench_action) >= 8:
            # Normalize quaternion part [qx, qy, qz, qw]
            quat = rlbench_action[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-8:  # Avoid division by zero
                rlbench_action[3:7] = quat / quat_norm
            else:
                # If quaternion is zero, use identity quaternion
                rlbench_action[3:7] = [0, 0, 0, 1]
        
        # Threshold gripper action: > 0.5 means open (1), <= 0.5 means close (0)
        if rlbench_action[-1] > 0.5:
            rlbench_action[-1] = 1.0
        else:
            rlbench_action[-1] = 0.0
        
        return rlbench_action
    
    def step(self, *args, **kwargs):
        """Execute one step in the environment."""
        # Extract action from MetaAction or dict
        if isinstance(args[0], MetaAction):
            action = args[0]['action']
        elif isinstance(args[0], dict):
            action = args[0]['action']
        else:
            action = args[0]
        
        # Convert to RLBench format
        rlbench_action = self.meta2act(MetaAction(action=action))
        
        # Execute step
        obs, reward, terminate = self.task.step(rlbench_action)
        
        # Convert observation
        meta_obs = self.obs2meta(obs)
        self.prev_obs = meta_obs
        
        # RLBench uses 'terminate' instead of 'done'
        # Create info dict
        info = {
            'success': terminate,
            'terminated': terminate,
            'truncated': False
        }
        
        return meta_obs, reward, terminate, info
    
    def reset(self):
        """Reset the environment and return initial observation."""
        # Get task if not already loaded
        if self.task is None:
            task_class = self._get_task_class(self.task_name)
            self.task = self.rlbench_env.get_task(task_class)
        
        # Reset task
        descriptions, obs = self.task.reset()
        
        # Store language descriptions
        self.descriptions = descriptions
        self.raw_lang = descriptions[0] if descriptions else ""
        
        # Convert observation
        meta_obs = self.obs2meta(obs)
        self.prev_obs = meta_obs
        
        return meta_obs
    
    def close(self):
        """Close the environment."""
        if self.rlbench_env is not None:
            self.rlbench_env.shutdown()
    
    def get_action_dim(self):
        """Return action dimension."""
        if self.ctrl_space == 'joint':
            return 8  # 7 joints + 1 gripper
        elif self.ctrl_space == 'ee':
            return 8  # 7 DoF pose (x,y,z,qx,qy,qz,qw) + 1 gripper
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")

