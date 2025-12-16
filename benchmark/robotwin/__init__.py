"""
RoboTwin Environment Integration for ILStudio

RoboTwin is a dual-arm manipulation environment with 50 tasks.
"""

import sys
import os
import numpy as np
import yaml
import importlib
from pathlib import Path

# Add RoboTwin to path
ROBOTWIN_ROOT = os.path.join(os.path.dirname(__file__), 'RoboTwin')
ROBOTWIN_ROOT = os.path.abspath(ROBOTWIN_ROOT)

# Change to RoboTwin directory for relative path resolution
_original_cwd = None

def _enter_robotwin_context():
    """Change to RoboTwin directory to resolve relative paths."""
    global _original_cwd
    _original_cwd = os.getcwd()
    os.chdir(ROBOTWIN_ROOT)

def _exit_robotwin_context():
    """Restore original working directory."""
    global _original_cwd
    if _original_cwd is not None:
        os.chdir(_original_cwd)

sys.path.insert(0, ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, 'policy'))
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, 'description/utils'))

from ..base import MetaEnv, MetaObs, MetaAction


def create_env(config):
    """
    Create RoboTwin environment instance.
    
    Args:
        config: Configuration object with task and settings
        
    Returns:
        RoboTwinEnv instance
    """
    return RoboTwinEnv(config)


class RoboTwinEnv(MetaEnv):
    """
    RoboTwin Environment wrapper for ILStudio.
    
    RoboTwin features:
    - Dual-arm manipulation tasks
    - Rich visual observations (RGB, depth, point cloud)
    - 50 different manipulation tasks
    - Domain randomization support
    """
    
    def __init__(self, config):
        """
        Initialize RoboTwin environment.
        
        Args:
            config: Configuration with attributes:
                - task: Task name (e.g., 'pick_diverse_bottles')
                - task_config: Config file name (e.g., 'demo_clean')
                - max_timesteps: Maximum episode steps
                - ctrl_space: 'qpos' or 'ee' (default: 'qpos')
                - ctrl_type: 'abs' (RoboTwin only supports absolute control)
                - seed: Random seed
                - image_size: [H, W] for cameras (default: [480, 640])
                - camera_names: List of cameras to use (default: ['head_camera'])
                - robotwin_root: Custom RoboTwin root path (optional)
        """
        self.config = config
        
        # Allow custom RoboTwin root path
        global ROBOTWIN_ROOT
        custom_root = getattr(config, 'robotwin_root', None)
        if custom_root:
            ROBOTWIN_ROOT = os.path.abspath(custom_root)
        
        self.robotwin_root = ROBOTWIN_ROOT
        self.task_name = config.task
        self.task_config_name = getattr(config, 'task_config', 'demo_clean')
        self.max_timesteps = config.max_timesteps
        self.ctrl_space = getattr(config, 'ctrl_space', 'qpos')
        self.ctrl_type = 'abs'  # RoboTwin only supports absolute control
        self.seed = getattr(config, 'seed', 0)
        self.image_size = getattr(config, 'image_size', [480, 640])
        self.camera_names = getattr(config, 'camera_names', ['head_camera'])
        
        # Robot embodiment configuration (optional override)
        # If not specified, will use embodiment from task_config YAML
        self.embodiment = getattr(config, 'embodiment', None)
        
        # Planner configuration
        # use_planner=False: Skip planner initialization (faster, no GPU required)
        # use_planner=True: Try to use TOPP/Curobo for trajectory smoothing (with automatic fallback)
        # Note: Original RoboTwin tries to use planner but gracefully falls back if it fails
        self.use_planner = getattr(config, 'use_planner', False)
        
        # Change to RoboTwin directory for initialization
        _enter_robotwin_context()
        
        try:
            # Load task config
            task_config_path = os.path.join(self.robotwin_root, 'task_config', f'{self.task_config_name}.yml')
            with open(task_config_path, 'r') as f:
                self.task_config = yaml.safe_load(f)
            
            # Override with ILStudio settings
            self.task_config['task_name'] = self.task_name
            self.task_config['eval_mode'] = True
            self.task_config['save_data'] = False
            self.task_config['render_freq'] = 0  # No rendering by default
            self.task_config['seed'] = self.seed
            self.task_config['use_planner'] = self.use_planner  # Motion planner (Curobo + TOPP)
            self.task_config['step_lim'] = self.max_timesteps  # Override RoboTwin's default step limit
            
            # Load camera config and set dimensions
            self._load_camera_configs()
            
            # Load embodiment config to get action dimensions
            self._load_embodiment_configs()
            
            # Import and create task environment
            self.task_env = self._create_task_env()
            
        finally:
            # Restore original directory
            _exit_robotwin_context()
        
        # Initialize episode counter
        self.episode_num = 0
        self._env_initialized = False
        
        # Calculate action dimension: (left_arm + left_gripper + right_arm + right_gripper)
        self.action_dim = self.left_arm_dim + 1 + self.right_arm_dim + 1
        
        # Create a dummy action_space for compatibility (gymnasium.spaces.Box)
        import gymnasium as gym
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Don't call parent init as we handle env differently
        self.env = None
        self.prev_obs = None
        
    def _load_camera_configs(self):
        """Load camera configurations and set dimensions."""
        # Use absolute path to RoboTwin's config directory
        configs_path = os.path.join(self.robotwin_root, 'task_config')
        
        # Load camera config
        camera_config_path = os.path.join(configs_path, "_camera_config.yml")
        with open(camera_config_path, 'r') as f:
            camera_configs = yaml.safe_load(f)
        
        # Get head camera type from task config
        if 'camera' not in self.task_config:
            self.task_config['camera'] = {}
        
        head_camera_type = self.task_config['camera'].get('head_camera_type', '480p')
        
        # Set camera dimensions based on camera type
        if head_camera_type in camera_configs:
            self.task_config['head_camera_h'] = camera_configs[head_camera_type]['h']
            self.task_config['head_camera_w'] = camera_configs[head_camera_type]['w']
        else:
            # Fallback to default
            self.task_config['head_camera_h'] = self.image_size[0]
            self.task_config['head_camera_w'] = self.image_size[1]
        
        # Override camera collection based on camera_names
        # Use RoboTwin's original camera names directly
        self.task_config['camera']['collect_head_camera'] = 'head_camera' in self.camera_names
        self.task_config['camera']['collect_wrist_camera'] = ('left_camera' in self.camera_names or 
                                                               'right_camera' in self.camera_names)
    
    def _load_embodiment_configs(self):
        """Load embodiment configurations to determine action dimensions."""
        # Use absolute path to RoboTwin's config directory
        configs_path = os.path.join(self.robotwin_root, 'task_config')
        
        # Load embodiment config
        embodiment_config_path = os.path.join(configs_path, "_embodiment_config.yml")
        with open(embodiment_config_path, 'r') as f:
            embodiment_types = yaml.safe_load(f)
        
        # Get embodiment type from ILStudio config (if specified) or task config
        if self.embodiment is not None:
            embodiment_type = self.embodiment if isinstance(self.embodiment, list) else [self.embodiment]
        else:
            embodiment_type = self.task_config.get('embodiment', ['aloha-agilex'])
        
        def get_embodiment_file(embodiment_name):
            """Get the robot file path for an embodiment."""
            robot_file = embodiment_types[embodiment_name]['file_path']
            if robot_file is None:
                raise ValueError(f"No file path for embodiment '{embodiment_name}'")
            return robot_file
        
        def get_embodiment_config(robot_file):
            """Load embodiment config from robot file."""
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Handle single or dual embodiment
        if len(embodiment_type) == 1:
            # Same robot for both arms (dual-arm embodied)
            left_robot_file = get_embodiment_file(embodiment_type[0])
            right_robot_file = left_robot_file
            self.task_config['dual_arm_embodied'] = True
        elif len(embodiment_type) == 3:
            # Different robots for each arm
            left_robot_file = get_embodiment_file(embodiment_type[0])
            right_robot_file = get_embodiment_file(embodiment_type[1])
            self.task_config['embodiment_dis'] = embodiment_type[2]
            self.task_config['dual_arm_embodied'] = False
        else:
            raise ValueError("embodiment_type must have 1 or 3 elements")
        
        # Set robot files in task config
        self.task_config['left_robot_file'] = left_robot_file
        self.task_config['right_robot_file'] = right_robot_file
        
        # Load embodiment configs
        left_config = get_embodiment_config(left_robot_file)
        right_config = get_embodiment_config(right_robot_file)
        
        self.task_config['left_embodiment_config'] = left_config
        self.task_config['right_embodiment_config'] = right_config
        
        # Get action dimensions
        self.left_arm_dim = len(left_config.get('arm_joints_name', [[]])[0])
        self.right_arm_dim = len(right_config.get('arm_joints_name', [[]])[1])
    
    def _create_task_env(self):
        """Dynamically load the task environment class."""
        try:
            envs_module = importlib.import_module(f"envs.{self.task_name}")
            env_class = getattr(envs_module, self.task_name)
            return env_class()
        except Exception as e:
            raise ValueError(f"Failed to load task '{self.task_name}': {e}")
    
    def _init_env_for_episode(self):
        """Initialize environment for a new episode."""
        # Update seed for this episode
        self.task_config['seed'] = self.seed + self.episode_num
        self.task_config['now_ep_num'] = self.episode_num
        
        # Change to RoboTwin directory for episode setup
        _enter_robotwin_context()
        
        try:
            # Setup the demo/episode
            self.task_env.setup_demo(is_test=True, **self.task_config)
            
            # Force override step_lim after setup to ensure our max_timesteps is used
            self.task_env.step_lim = self.max_timesteps
            
            self._env_initialized = True
        except Exception as e:
            # Some episodes may fail due to unstable object placement
            # This is normal in RoboTwin
            self._env_initialized = False
            raise RuntimeError(f"Failed to initialize episode {self.episode_num}: {e}")
        finally:
            # Restore original directory
            _exit_robotwin_context()
    
    def reset(self):
        """
        Reset environment and return initial observation.
        
        Returns:
            MetaObs: Initial observation in ILStudio format
        """
        # Close previous environment if exists
        if self._env_initialized:
            try:
                self.task_env.close_env()
            except:
                pass
        
        # Try to initialize new episode (may need multiple attempts)
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                self._init_env_for_episode()
                break
            except Exception as e:
                self.episode_num += 1
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to initialize environment after {max_attempts} attempts")
        
        # Get initial observation
        obs_dict = self.task_env.get_obs()
        
        # Convert to MetaObs
        meta_obs = self.obs2meta(obs_dict)
        self.prev_obs = meta_obs
        
        # Reset step counter
        self.task_env.take_action_cnt = 0
        
        return meta_obs
    
    def obs2meta(self, obs_dict):
        """
        Convert RoboTwin observation to MetaObs format.
        
        RoboTwin obs structure:
        {
            'observation': {
                'head_camera': {'rgb': (H, W, 3)},
                'left_wrist_camera': {'rgb': (H, W, 3)},
                'right_wrist_camera': {'rgb': (H, W, 3)},
            },
            'joint_action': {
                'left_arm': (N,),
                'left_gripper': float,
                'right_arm': (N,),
                'right_gripper': float,
                'vector': (2N+2,)
            },
            'endpose': {
                'left_endpose': (7,) [x,y,z,qx,qy,qz,qw],
                'left_gripper': float,
                'right_endpose': (7,),
                'right_gripper': float
            }
        }
        
        Returns:
            MetaObs with state and images
        """
        # Extract images
        images = []
        observation = obs_dict.get('observation', {})
        
        for cam_name in self.camera_names:
            # Directly use camera name from config
            # RoboTwin's camera keys: 'head_camera', 'left_camera', 'right_camera'
            if cam_name in observation and 'rgb' in observation[cam_name]:
                img = observation[cam_name]['rgb']  # (H, W, 3) RGB
                # Convert to (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
        
        if len(images) > 0:
            image = np.stack(images)  # (N, C, H, W)
        else:
            image = None
        
        # Extract state based on control space
        if self.ctrl_space == 'qpos':
            # Use joint positions as state
            joint_action = obs_dict.get('joint_action', {})
            if 'vector' in joint_action:
                state = joint_action['vector'].astype(np.float32)
            else:
                # Fallback: concatenate left and right
                left_arm = joint_action.get('left_arm', np.array([]))
                left_gripper = np.array([joint_action.get('left_gripper', 0.0)])
                right_arm = joint_action.get('right_arm', np.array([]))
                right_gripper = np.array([joint_action.get('right_gripper', 0.0)])
                state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)
        elif self.ctrl_space == 'ee':
            # Use end-effector poses as state
            endpose = obs_dict.get('endpose', {})
            left_endpose = endpose.get('left_endpose', np.zeros(7))
            left_gripper = np.array([endpose.get('left_gripper', 0.0)])
            right_endpose = endpose.get('right_endpose', np.zeros(7))
            right_gripper = np.array([endpose.get('right_gripper', 0.0)])
            state = np.concatenate([left_endpose, left_gripper, right_endpose, right_gripper]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")
        
        return MetaObs(
            state=state,
            state_joint=obs_dict.get('joint_action', {}).get('vector', None),
            state_ee=None,  # RoboTwin doesn't separate ee state cleanly
            image=image,
            raw_lang=getattr(self.task_env, 'instruction', '')
        )
    
    def meta2act(self, maction: MetaAction):
        """
        Convert MetaAction to RoboTwin action format.
        
        RoboTwin expects action as numpy array:
        - For qpos: [left_arm_joints, left_gripper, right_arm_joints, right_gripper]
        - For ee: [left_ee_pose(7), left_gripper, right_ee_pose(7), right_gripper]
        
        Args:
            maction: MetaAction with action array
            
        Returns:
            action: numpy array for RoboTwin
        """
        action = maction['action']
        
        # Ensure action is float32
        action = np.array(action, dtype=np.float32)
        
        return action
    
    def step(self, *args, **kwargs):
        """
        Execute one step in the environment.
        
        Args:
            maction: MetaAction or dict with 'action' key
            
        Returns:
            tuple: (obs, reward, done, info)
        """
        # Extract action
        if isinstance(args[0], MetaAction):
            action = args[0]['action']
        elif isinstance(args[0], dict):
            action = args[0]['action']
        else:
            action = args[0]
        
        # Convert to RoboTwin format
        robotwin_action = self.meta2act(MetaAction(action=action))
        
        # Change to RoboTwin directory for step execution
        _enter_robotwin_context()
        
        try:
            # Execute action
            action_type = self.ctrl_space  # 'qpos' or 'ee'
            self.task_env.take_action(robotwin_action, action_type=action_type)
            
            # Get new observation
            obs_dict = self.task_env.get_obs()
        finally:
            # Restore original directory
            _exit_robotwin_context()
        
        meta_obs = self.obs2meta(obs_dict)
        self.prev_obs = meta_obs
        
        # Check success
        success = self.task_env.eval_success
        
        # In ILStudio, done indicates task success, not episode termination
        # This matches the convention used in other environments (simplerenv, gymnasium_robotics, etc.)
        done = success
        
        # Check termination conditions
        terminated = (self.task_env.take_action_cnt >= self.max_timesteps) or success
        truncated = self.task_env.take_action_cnt >= self.max_timesteps and not success
        
        # Reward (0 or 1 based on success)
        reward = float(success)
        
        # Info
        info = {
            'success': success,
            'terminated': terminated,
            'truncated': truncated,
            'step_count': self.task_env.take_action_cnt
        }
        
        return meta_obs, reward, done, info
    
    def close(self):
        """Close the environment and cleanup."""
        if self._env_initialized:
            _enter_robotwin_context()
            try:
                self.task_env.close_env()
            except:
                pass
            finally:
                _exit_robotwin_context()
        self._env_initialized = False
    
    def get_action_dim(self):
        """Return action dimension based on control space."""
        # Action dim depends on the robot configuration
        # For dual-arm ALOHA-Agilex: 7+1 joints per arm = 16 total for qpos
        # For ee control: 7+1 per arm = 16 total
        
        if self.ctrl_space == 'qpos':
            # Joint space: number of joints + gripper per arm
            return self.left_arm_dim + 1 + self.right_arm_dim + 1
        elif self.ctrl_space == 'ee':
            # End-effector: 7-DoF pose + gripper per arm
            return 7 + 1 + 7 + 1  # Always 16 for ee mode
        else:
            raise ValueError(f"Unsupported ctrl_space: {self.ctrl_space}")

