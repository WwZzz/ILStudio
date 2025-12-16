"""
Test script for RoboTwin integration with ILStudio.

This script validates that RoboTwin can be loaded and used with ILStudio.

Usage:
    benchmark/robotwin/.venv/bin/python benchmark/robotwin/test_integration.py
"""

import sys
import os

# Add ILStudio root to path
ILSTUDIO_ROOT = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, ILSTUDIO_ROOT)

import numpy as np
from configs.loader import ConfigLoader
import argparse

print("=" * 70)
print("Testing RoboTwin Integration with ILStudio")
print("=" * 70)

# Parse config
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='robotwin_pick_bottles')
args, unknown = parser.parse_known_args()

print(f"\n[1/6] Loading configuration...")
cfg_loader = ConfigLoader(args=args, unknown_args=unknown)
env_cfg, cfg_path = cfg_loader.load_env(args.env)

# Handle list config
if isinstance(env_cfg, list):
    env_cfg = env_cfg[0]

print(f"✓ Config loaded from: {cfg_path}")
print(f"  - Type: {env_cfg.type}")
print(f"  - Name: {env_cfg.name}")
print(f"  - Task: {env_cfg.task}")
print(f"  - Max timesteps: {env_cfg.max_timesteps}")
print(f"  - Control space: {env_cfg.ctrl_space}")

# Import create_env
print(f"\n[2/6] Importing environment...")
from benchmark.robotwin import create_env
print("✓ Environment module imported")

# Create environment
print(f"\n[3/6] Creating environment...")
try:
    env = create_env(env_cfg)
    print("✓ Environment created successfully")
    print(f"  - Action dim: {env.get_action_dim()}")
    print(f"  - Left arm joints: {env.left_arm_dim}")
    print(f"  - Right arm joints: {env.right_arm_dim}")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Reset environment
print(f"\n[4/6] Resetting environment...")
try:
    obs = env.reset()
    print("✓ Environment reset successfully")
    print(f"  - State shape: {obs.state.shape}")
    if obs.image is not None:
        print(f"  - Image shape: {obs.image.shape}")
    print(f"  - Language: '{obs.raw_lang}'")
except Exception as e:
    print(f"✗ Failed to reset: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

# Take a step
print(f"\n[5/6] Testing step execution...")
try:
    from benchmark.base import MetaAction
    action_dim = env.get_action_dim()
    
    # Create random action
    action = np.random.uniform(-0.1, 0.1, size=action_dim).astype(np.float32)
    # Set grippers to reasonable values
    action[env.left_arm_dim] = 0.5  # left gripper
    action[env.left_arm_dim + env.right_arm_dim + 1] = 0.5  # right gripper
    
    meta_action = MetaAction(action=action, ctrl_space=env.ctrl_space)
    obs, reward, done, info = env.step(meta_action)
    
    print("✓ Step executed successfully")
    print(f"  - Reward: {reward}")
    print(f"  - Done: {done}")
    print(f"  - Success: {info['success']}")
    print(f"  - Step count: {info['step_count']}")
except Exception as e:
    print(f"✗ Failed to step: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    sys.exit(1)

# Close environment
print(f"\n[6/6] Closing environment...")
try:
    env.close()
    print("✓ Environment closed successfully")
except Exception as e:
    print(f"⚠ Warning during close: {e}")

print("\n" + "=" * 70)
print("✅ All tests passed! RoboTwin is integrated with ILStudio!")
print("=" * 70)
print("\nUsage:")
print("  benchmark/robotwin/.venv/bin/python eval_sim.py \\")
print(f"    -e {args.env} -m <model> --batch_size 0")
print("\nRemember: Always use --batch_size 0 for RoboTwin!")

