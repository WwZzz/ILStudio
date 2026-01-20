"""
Test script for rollout and replay buffer verification.

This script:
1. Loads a trained policy model
2. Performs rollouts in parallel environments
3. Adds rollout data to a replay buffer (handling multiple envs)
4. Samples complete episodes from the replay buffer
5. Saves observation images as video for verification

Usage:
    python test_rollout.py -m /path/to/checkpoint -e aloha -n 4 -bs 2 -o results/test_rollout
"""

import configs  # Must be first to suppress TensorFlow logs
import os
import json
import numpy as np
import multiprocessing as mp
from dataclasses import asdict
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any
from tianshou.env import SubprocVectorEnv

from data_utils.utils import set_seed
from benchmark.utils import SequentialVectorEnv, organize_obs
from policy.utils import load_policy
from policy.rl.replay_buffer import RolloutReplayBuffer


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test rollout and replay buffer')
    parser.add_argument('-o', '--output_dir', type=str, default='results/test_rollout',
                       help='Directory to save results')
    parser.add_argument('-s', '--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('-m', '--model_name_or_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_id', type=str, default='',
                       help='Dataset ID (for local model loading)')
    parser.add_argument('-e', '--env', type=str, default='aloha',
                       help='Env config name or path')
    parser.add_argument('--fps', type=int, default=50,
                       help='Video FPS')
    parser.add_argument('-n', '--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('-bs', '--batch_size', type=int, default=2,
                       help='Number of parallel environments')
    parser.add_argument('--use_spawn', action='store_true',
                       help='Use spawn method for multiprocessing')
    parser.add_argument('-ck', '--chunk_size', type=int, default=-1,
                       help='Action chunk size')
    parser.add_argument('--ctrl_space', type=str, default='ee',
                       help='Control space (ee or joint)')
    parser.add_argument('--ctrl_type', type=str, default='delta',
                       help='Control type (delta or abs)')
    
    args, unknown = parser.parse_known_args()
    args._unknown = unknown
    return args


def metaobs_to_dict(obs) -> Dict[str, np.ndarray]:
    """Convert MetaObs or dict observation to standard dict format."""
    if isinstance(obs, dict):
        return obs
    elif hasattr(obs, '__dataclass_fields__'):
        return asdict(obs)
    else:
        raise TypeError(f"Unknown observation type: {type(obs)}")


def perform_rollout(
    args,
    policy,
    env,
    replay_buffer: RolloutReplayBuffer,
) -> Dict[str, Any]:
    """
    Perform one rollout iteration and add data to replay buffer.
    
    Args:
        args: Command line arguments
        policy: The policy model
        env: Vectorized environment
        replay_buffer: Buffer to store transitions
        
    Returns:
        Dict with rollout statistics
    """
    num_envs = len(env)
    horizons = np.ones(num_envs) * args.max_timesteps
    success = np.zeros(num_envs, dtype=bool)
    total_rewards = np.zeros(num_envs)
    env_done = np.zeros(num_envs, dtype=bool)  # Track which envs have finished
    
    # Reset environments
    obs = env.reset()
    obs_dict = metaobs_to_dict(organize_obs(obs, args.ctrl_space))
    
    for t in range(args.max_timesteps):
        # Get action from policy
        obs_for_policy = organize_obs(obs, args.ctrl_space)
        act = policy.select_action(obs_for_policy, t)
        
        # Extract action array from policy output
        if isinstance(act, np.ndarray) and act.dtype == np.object_:
            # Array of MetaAction/dicts
            action_batch = np.array([
                a['action'] if isinstance(a, dict) else a.action 
                for a in act
            ])
        elif hasattr(act, 'action'):
            action_batch = act.action  # Single MetaAction
            if action_batch.ndim == 1:
                action_batch = np.expand_dims(action_batch, 0)
        else:
            action_batch = np.array(act)
            if action_batch.ndim == 1:
                action_batch = np.expand_dims(action_batch, 0)
        
        # Step environment
        next_obs_raw, reward, done, info = env.step(act)
        next_obs_dict = metaobs_to_dict(organize_obs(next_obs_raw, args.ctrl_space))
        
        # Add to replay buffer (skip already-done environments to avoid
        # adding data from auto-reset episodes)
        replay_buffer.add_from_parallel_envs(
            obs_batch=obs_dict,
            action_batch=action_batch,
            reward_batch=np.array(reward),
            next_obs_batch=next_obs_dict,
            done_batch=np.array(done),
            info_batch=info,
            skip_mask=env_done,  # Skip envs that already finished
        )
        
        # Update statistics (only for active envs)
        active_mask = ~env_done
        total_rewards[active_mask] += np.array(reward)[active_mask]
        
        # Track newly done environments
        newly_done = np.array(done) & ~env_done
        success = success | newly_done
        
        # Update horizon for successful episodes
        for idx in range(num_envs):
            if newly_done[idx] and horizons[idx] > t:
                horizons[idx] = t
        
        # Mark environments as done AFTER adding their final transition
        env_done = env_done | np.array(done)
        
        # Check if all done
        if env_done.all():
            break
        
        # Move to next state
        obs = next_obs_raw
        obs_dict = next_obs_dict
    
    return {
        'success': success.tolist(),
        'total_success': int(success.sum()),
        'total': num_envs,
        'horizons': horizons.tolist(),
        'total_rewards': total_rewards.tolist(),
        'avg_reward': float(total_rewards.mean()),
    }


def main():
    args = parse_args()
    args.is_training = False
    set_seed(args.seed)
    
    if args.use_spawn:
        mp.set_start_method('spawn', force=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =========================================================================
    # Load policy
    # =========================================================================
    logger.info("="*60)
    logger.info("Loading policy...")
    logger.info("="*60)
    policy = load_policy(args)
    
    # =========================================================================
    # Load environment config
    # =========================================================================
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, '_unknown', []))
    env_cfg, env_cfg_path = cfg_loader.load_env(args.env)
    
    # Handle multiple env configs
    if isinstance(env_cfg, list):
        env_cfg = env_cfg[0]  # Use first config for testing
    
    # Get task and max_timesteps from env config
    if hasattr(env_cfg, 'task'):
        args.task = env_cfg.task
    if hasattr(env_cfg, 'max_timesteps'):
        args.max_timesteps = env_cfg.max_timesteps
    if hasattr(env_cfg, 'ctrl_space'):
        args.ctrl_space = env_cfg.ctrl_space
    
    # Load environment module
    import importlib
    env_type = env_cfg.type
    if '.' in env_type:
        module_path, class_name = env_type.rsplit('.', 1)
        env_module = importlib.import_module(module_path)
        env_name = module_path.split('.')[-1]
    else:
        env_module = importlib.import_module(f"benchmark.{env_type}")
        env_name = env_type
    
    if not hasattr(env_module, 'create_env'):
        raise AttributeError(f"Env module {env_type} has no 'create_env'")
    
    logger.info(f"Environment: {env_name}, Task: {args.task}")
    logger.info(f"Max timesteps: {args.max_timesteps}")
    logger.info(f"Control: {args.ctrl_space}/{args.ctrl_type}")
    
    # =========================================================================
    # Create replay buffer
    # =========================================================================
    # Estimate capacity: num_rollouts * max_timesteps
    estimated_capacity = args.num_rollout * args.max_timesteps
    replay_buffer = RolloutReplayBuffer(
        capacity=estimated_capacity,
        num_envs=max(args.batch_size, 1),
        chunk_size=1,  # Store single-step actions
        ctrl_space=args.ctrl_space,
        ctrl_type=args.ctrl_type,
        device=args.device,
        storage_device='cpu',
    )
    
    logger.info(f"Created replay buffer with capacity: {estimated_capacity}")
    
    # =========================================================================
    # Perform rollouts
    # =========================================================================
    logger.info("="*60)
    logger.info("Performing rollouts...")
    logger.info("="*60)
    
    def env_fn(env_config, env_handler):
        def create_env():
            return env_handler(env_config)
        return create_env
    
    all_results = []
    use_sequential = (args.batch_size == 0)
    batch_size = 1 if use_sequential else args.batch_size
    num_iters = args.num_rollout if use_sequential else (args.num_rollout + batch_size - 1) // batch_size

    for i in tqdm(range(num_iters), desc="Rollouts"):
        num_envs = 1 if use_sequential else min(batch_size, args.num_rollout - i * batch_size)
        
        # Create environments
        env_fns = [env_fn(env_cfg, env_module.create_env) for _ in range(num_envs)]
        env = SequentialVectorEnv(env_fns) if use_sequential else SubprocVectorEnv(env_fns)
        
        # Set policy to eval mode
        if hasattr(policy, 'policy') and hasattr(policy.policy, 'eval'):
            policy.policy.eval()
        
        # Perform rollout
        result = perform_rollout(args, policy, env, replay_buffer)
        all_results.append(result)
        logger.info(f"Rollout {i}: success={result['total_success']}/{result['total']}, avg_reward={result['avg_reward']:.2f}")
        
        # Reset policy
        policy.reset()
        env.close()
    
    # =========================================================================
    # Aggregate results
    # =========================================================================
    total_success = sum(r['total_success'] for r in all_results)
    total_rollouts = sum(r['total'] for r in all_results)
    avg_reward = np.mean([r['avg_reward'] for r in all_results])
    
    logger.info("="*60)
    logger.info("Rollout Summary")
    logger.info("="*60)
    logger.info(f"Success rate: {total_success}/{total_rollouts} = {100*total_success/total_rollouts:.1f}%")
    logger.info(f"Average reward: {avg_reward:.3f}")
    logger.info(f"Replay buffer size: {replay_buffer.size}")
    
    # =========================================================================
    # Save replay buffer statistics
    # =========================================================================
    episode_ids = replay_buffer.get_all_episode_ids()
    episode_lengths = replay_buffer.get_episode_lengths()
    
    logger.info(f"Number of episodes: {len(episode_ids)}")
    logger.info(f"Episode lengths: min={min(episode_lengths.values())}, max={max(episode_lengths.values())}, avg={np.mean(list(episode_lengths.values())):.1f}")
    
    # Save summary (convert numpy types to Python native types for JSON serialization)
    summary = {
        'total_success': int(total_success),
        'total_rollouts': int(total_rollouts),
        'success_rate': float(total_success / total_rollouts),
        'avg_reward': float(avg_reward),
        'num_episodes': len(episode_ids),
        'replay_buffer_size': int(replay_buffer.size),
        'episode_ids': [int(x) for x in episode_ids],
        'episode_lengths': {str(k): int(v) for k, v in episode_lengths.items()},
        'all_results': all_results,
    }
    
    summary_path = os.path.join(args.output_dir, 'rollout_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary to {summary_path}")
    
    # =========================================================================
    # Sample complete episodes and save as videos
    # =========================================================================
    logger.info("="*60)
    logger.info("Saving episode videos for verification...")
    logger.info("="*60)
    
    video_dir = os.path.join(args.output_dir, 'episode_videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Save videos for a few episodes
    num_videos = min(5, len(episode_ids))
    for i, ep_id in enumerate(episode_ids[:num_videos]):
        video_path = os.path.join(video_dir, f'episode_{ep_id}.mp4')
        
        # Use buffer's save_episode_as_video method
        replay_buffer.save_episode_as_video(ep_id, video_path, fps=args.fps)
        
        # Also save episode data as numpy for debugging
        episode_data = replay_buffer.get_episode_by_id(ep_id)
        data_path = os.path.join(video_dir, f'episode_{ep_id}_data.npz')
        
        # Extract data arrays
        if len(episode_data) > 0:
            actions = np.array([t['action'] for t in episode_data])
            rewards = np.array([t['reward'] for t in episode_data])
            dones = np.array([t['done'] for t in episode_data])
            timestamps = np.array([t['timestamp'] for t in episode_data])
            
            np.savez(data_path, 
                    actions=actions, 
                    rewards=rewards, 
                    dones=dones,
                    timestamps=timestamps,
                    episode_id=ep_id,
                    episode_length=len(episode_data))
            logger.info(f"Saved episode data to {data_path}")
    
    # =========================================================================
    # Verify sampled data structure
    # =========================================================================
    logger.info("="*60)
    logger.info("Verifying sampled data structure...")
    logger.info("="*60)
    
    # Sample a few transitions to verify structure
    if len(episode_ids) > 0:
        ep_id = episode_ids[0]
        episode_data = replay_buffer.get_episode_by_id(ep_id)
        
        if len(episode_data) > 0:
            sample = episode_data[0]
            logger.info(f"Sample transition from episode {ep_id}:")
            logger.info(f"  Observation keys: {list(sample['obs'].keys())}")
            for key, value in sample['obs'].items():
                if isinstance(value, np.ndarray):
                    logger.info(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"    {key}: {type(value).__name__}")
            logger.info(f"  Action: shape={sample['action'].shape}, dtype={sample['action'].dtype}")
            logger.info(f"  Reward: {sample['reward']}")
            logger.info(f"  Done: {sample['done']}")
            logger.info(f"  Timestamp: {sample['timestamp']}")
            logger.info(f"  Episode ID: {sample['episode_id']}")
    
    logger.info("="*60)
    logger.info("Test complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

