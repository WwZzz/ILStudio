"""
DeepMind Control Suite (DMC) Environment Wrapper for ILStudio.

This module provides:
1. DMC environment creation with image observations
2. Frame stacking for temporal information
3. Action repeat for sample efficiency
4. Compatible with ILStudio's benchmark interface (inherits MetaEnv)

Note: Uses dm_control directly instead of dmc2gym for better compatibility.
"""

import os
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Tuple
from loguru import logger

try:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    DM_CONTROL_AVAILABLE = True
except ImportError:
    DM_CONTROL_AVAILABLE = False
    print("Warning: dm_control not installed. Install with: pip install dm_control")

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from ..base import MetaEnv, MetaObs, MetaAction


# ============================================================================
# DMC to Gym Wrapper
# ============================================================================

class DMCWrapper(gym.Env):
    """
    Wrapper to convert dm_control environment to gym interface.
    
    This replaces dmc2gym for better compatibility with modern gym/gymnasium.
    """
    
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        seed: int = 0,
        from_pixels: bool = True,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        channels_first: bool = True,
    ):
        self._domain_name = domain_name
        self._task_name = task_name
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        
        # Create dm_control environment
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={'random': seed},
        )
        
        # Wrap with pixel observations if needed
        if from_pixels:
            self._env = pixels.Wrapper(
                self._env,
                pixels_only=True,
                render_kwargs={
                    'height': height,
                    'width': width,
                    'camera_id': camera_id,
                },
            )
        
        # Get action spec
        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        
        # Set observation space
        if from_pixels:
            if channels_first:
                shape = (3, height, width)
            else:
                shape = (height, width, 3)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=shape,
                dtype=np.uint8,
            )
        else:
            # State-based observation
            obs_spec = self._env.observation_spec()
            obs_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        
        self._max_episode_steps = 1000  # Default for most DMC tasks
        self._seed = seed
    
    def seed(self, seed=None):
        """Set random seed."""
        self._seed = seed
        return [seed]
    
    def reset(self, **kwargs):
        """Reset environment."""
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs, {}
    
    def step(self, action):
        """Execute action."""
        reward = 0.0
        
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break
        
        obs = self._get_obs(time_step)
        done = time_step.last()
        truncated = False
        info = {}
        
        return obs, reward, done, truncated, info
    
    def _get_obs(self, time_step):
        """Extract observation from time step."""
        if self._from_pixels:
            obs = time_step.observation['pixels']
            if self._channels_first:
                obs = obs.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            return obs.copy()
        else:
            # Concatenate all observations
            obs_list = []
            for key, value in time_step.observation.items():
                obs_list.append(value.flatten())
            return np.concatenate(obs_list).astype(np.float32)
    
    def render(self, mode='rgb_array'):
        """Render environment."""
        if mode == 'rgb_array':
            return self._env.physics.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id,
            )
        return None
    
    def close(self):
        """Close environment."""
        if hasattr(self._env, 'close'):
            self._env.close()


# ============================================================================
# Environment Wrappers
# ============================================================================

class FrameStack(gym.Wrapper):
    """
    Stack k consecutive frames as observation.
    """
    def __init__(self, env, k: int):
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=np.uint8
        )
        self._max_episode_steps = getattr(env, '_max_episode_steps', 1000)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class NormalizeActions(gym.Wrapper):
    """Normalize action space to [-1, 1]."""
    def __init__(self, env):
        super().__init__(env)
        self._low = env.action_space.low
        self._high = env.action_space.high
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )
        self._max_episode_steps = getattr(env, '_max_episode_steps', 1000)

    def step(self, action):
        # Denormalize action from [-1, 1] to [low, high]
        action = (action + 1.0) * (self._high - self._low) / 2.0 + self._low
        return self.env.step(action)


# ============================================================================
# DMC Environment (Inherits MetaEnv)
# ============================================================================

def create_env(config):
    """Create DMC environment from config (ILStudio interface)."""
    return DMCEnv(config)


class DMCEnv(MetaEnv):
    """
    DMC Environment wrapper that inherits from MetaEnv.
    """
    
    def __init__(self, config, *args):
        self.config = config
        self.ctrl_space = getattr(config, 'ctrl_space', 'joint')
        self.ctrl_type = getattr(config, 'ctrl_type', 'abs')
        self.use_camera = getattr(config, 'use_camera', True)
        self.render_mode = getattr(config, 'render_mode', 'rgb_array')
        
        # DMC specific
        self.image_size = getattr(config, 'image_size', 84)
        self.action_repeat = getattr(config, 'action_repeat', 2)
        self.frame_stack = getattr(config, 'frame_stack', 3)
        self.seed_val = getattr(config, 'seed', 0)
        self.normalize_actions = getattr(config, 'normalize_actions', True)
        
        # Task description
        self.raw_lang = get_task_description(config.task)
        
        # Create environment
        env = self.create_env()
        super().__init__(env)
        
        # Expose action_space and observation_space from underlying env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def create_env(self):
        """Create the underlying DMC environment with wrappers."""
        if not DM_CONTROL_AVAILABLE:
            raise ImportError("dm_control not installed. Install with: pip install dm_control")
        
        task = self.config.task
        domain_name, task_name = parse_dmc_env_name(task)
        
        # Use camera_id=2 for quadruped (better viewpoint)
        camera_id = 2 if domain_name == 'quadruped' else 0
        
        # Create base environment using our DMCWrapper
        env = DMCWrapper(
            domain_name=domain_name,
            task_name=task_name,
            seed=self.seed_val,
            from_pixels=True,
            height=self.image_size,
            width=self.image_size,
            camera_id=camera_id,
            frame_skip=self.action_repeat,
            channels_first=True,
        )
        
        # Apply frame stacking
        if self.frame_stack > 1:
            env = FrameStack(env, k=self.frame_stack)
        
        # Normalize actions to [-1, 1]
        if self.normalize_actions:
            env = NormalizeActions(env)
        
        # Get action bounds
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        self._max_episode_steps = getattr(env, '_max_episode_steps', 1000)
        
        logger.info(f"Created DMC environment: {self.config.task}")
        logger.info(f"  Observation space: {env.observation_space.shape}")
        logger.info(f"  Action space: {env.action_space.shape}")
        logger.info(f"  Action range: [{self.min_action.min()}, {self.max_action.max()}]")
        
        return env

    def meta2act(self, maction: MetaAction):
        """Convert MetaAction to environment action."""
        action = maction['action']
        return action
    
    def obs2meta(self, obs) -> MetaObs:
        """Convert raw observation to MetaObs."""
        if self.use_camera:
            if isinstance(obs, np.ndarray):
                if obs.ndim == 3:  # (C*k, H, W)
                    image = obs[np.newaxis, ...]  # (1, C*k, H, W)
                else:
                    image = obs
            else:
                image = obs
        else:
            image = None
        
        state = np.zeros(1, dtype=np.float32)
        
        return MetaObs(
            state=state,
            image=image,
            raw_lang=self.raw_lang,
        )

    def step(self, *args, **kwargs):
        """Execute action and return (obs, reward, done, info)."""
        if len(args) > 0:
            action = args[0]
            if isinstance(action, dict):
                action = action['action']
            elif hasattr(action, 'action'):
                action = action.action
        else:
            action = kwargs.get('action')
        
        obs, reward, done, truncated, info = self.env.step(action)
        meta_obs = self.obs2meta(obs)
        
        info['terminated'] = done
        info['truncated'] = truncated
        info['success'] = done and not truncated
        
        return meta_obs, reward, done, info
    
    def reset(self):
        """Reset environment and return initial observation."""
        obs, info = self.env.reset()
        return self.obs2meta(obs)
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]
    
    def render(self, mode='rgb_array'):
        """Render environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None
    
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


# ============================================================================
# Utility Functions
# ============================================================================

def parse_dmc_env_name(env_name: str) -> Tuple[str, str]:
    """Parse DMC environment name into domain and task."""
    special_cases = {
        'ball_in_cup_catch': ('ball_in_cup', 'catch'),
        'point_mass_easy': ('point_mass', 'easy'),
        'point_mass_hard': ('point_mass', 'hard'),
    }
    
    if env_name in special_cases:
        return special_cases[env_name]
    
    parts = env_name.split('_')
    domain_name = parts[0]
    task_name = '_'.join(parts[1:])
    return domain_name, task_name


def make_dmc_env(
    env_name: str,
    seed: int = 0,
    image_size: int = 84,
    action_repeat: int = 2,
    frame_stack: int = 3,
    normalize_actions: bool = True,
) -> gym.Env:
    """
    Create a raw DMC environment with standard wrappers (not MetaEnv).
    """
    if not DM_CONTROL_AVAILABLE:
        raise ImportError("dm_control not installed. Install with: pip install dm_control")
    
    domain_name, task_name = parse_dmc_env_name(env_name)
    camera_id = 2 if domain_name == 'quadruped' else 0
    
    env = DMCWrapper(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        from_pixels=True,
        height=image_size,
        width=image_size,
        camera_id=camera_id,
        frame_skip=action_repeat,
        channels_first=True,
    )
    
    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)
    
    if normalize_actions:
        env = NormalizeActions(env)
    
    return env


# ============================================================================
# Available Tasks
# ============================================================================

DMC_TASKS = {
    'cartpole_balance': 'Balance a pole on a cart',
    'cartpole_balance_sparse': 'Balance a pole on a cart (sparse reward)',
    'cartpole_swingup': 'Swing up and balance a pole',
    'cartpole_swingup_sparse': 'Swing up and balance a pole (sparse reward)',
    'cheetah_run': 'Make a cheetah run',
    'walker_stand': 'Make a walker stand',
    'walker_walk': 'Make a walker walk',
    'walker_run': 'Make a walker run',
    'hopper_stand': 'Make a hopper stand',
    'hopper_hop': 'Make a hopper hop',
    'finger_spin': 'Spin a finger',
    'finger_turn_easy': 'Turn a finger (easy)',
    'finger_turn_hard': 'Turn a finger (hard)',
    'ball_in_cup_catch': 'Catch a ball in a cup',
    'reacher_easy': 'Reach a target (easy)',
    'reacher_hard': 'Reach a target (hard)',
    'quadruped_walk': 'Make a quadruped walk',
    'quadruped_run': 'Make a quadruped run',
    'point_mass_easy': 'Move a point mass to target (easy)',
    'point_mass_hard': 'Move a point mass to target (hard)',
    'pendulum_swingup': 'Swing up a pendulum',
    'acrobot_swingup': 'Swing up an acrobot',
    'acrobot_swingup_sparse': 'Swing up an acrobot (sparse reward)',
    'humanoid_stand': 'Make a humanoid stand',
    'humanoid_walk': 'Make a humanoid walk',
    'humanoid_run': 'Make a humanoid run',
}


def list_tasks():
    """List all available DMC tasks."""
    return list(DMC_TASKS.keys())


def get_task_description(task: str) -> str:
    """Get description for a task."""
    return DMC_TASKS.get(task, f"DMC task: {task}")
