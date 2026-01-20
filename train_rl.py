"""
RL Training Script for ILStudio

Supports both online RL (e.g., DrQ) and offline RL (e.g., IQL) with dynamic loading.

Usage:
    # Online RL (DrQ)
    python train_rl.py -p drq -e dmc/cartpole_swingup -o ckpt/drq_cartpole
    
    # Offline RL (IQL)
    python train_rl.py -p iql -t d4rl/halfcheetah-medium -o ckpt/iql_halfcheetah
"""

import configs
import os
import argparse
import json
from loguru import logger
from data_utils.utils import set_seed
from configs.loader import ConfigLoader
from policy.policy_loader import (
    get_policy_trainer_class,
    load_policy_model_for_training,
)
import torch
import numpy as np
try:
    torch.serialization.add_safe_globals([np.ndarray])
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    torch.serialization.add_safe_globals([np.dtype])
    torch.serialization.safe_globals([np.dtype])
except:
    pass


def parse_param():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL policy')
    parser.add_argument('-p', '--policy', type=str, required=True,
                       help='Policy config (name under configs/policy or absolute path)')
    parser.add_argument('-e', '--env', type=str, default=None,
                       help='Env config for online RL (name under configs/env or absolute path)')
    parser.add_argument('-t', '--task', type=str, default=None,
                       help='Task config for offline RL (name under configs/task or absolute path)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    
    args, unknown = parser.parse_known_args()
    setattr(args, 'unknown_args', unknown)
    return args


def load_configs(args):
    """Load policy and env/task configs."""
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, 'unknown_args', []))
    
    # Load policy config
    policy_config, policy_cfg_path = cfg_loader.load_policy(args.policy)
    
    # Load env or task config based on policy type
    policy_type = policy_config.get('type', '')
    env_config = None
    task_config = None
    
    if policy_type.startswith('policy.rl.online.'):
        # Online RL: need env config
        if args.env is None:
            raise ValueError(f"Online RL policy {policy_type} requires -e/--env argument")
        env_ns, env_cfg_path = cfg_loader.load_env(args.env)
        logger.info(f"Loaded env config: {env_cfg_path}")
        # Convert namespace to dict for easier access
        from types import SimpleNamespace
        env_config = vars(env_ns) if isinstance(env_ns, SimpleNamespace) else env_ns
    elif policy_type.startswith('policy.rl.offline.'):
        # Offline RL: need task config
        if args.task is None:
            raise ValueError(f"Offline RL policy {policy_type} requires -t/--task argument")
        task_config, task_cfg_path = cfg_loader.load_task(args.task)
        logger.info(f"Loaded task config: {task_cfg_path}")
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Expected policy.rl.online.* or policy.rl.offline.*")
    
    # Merge parameters
    if env_config:
        ConfigLoader.merge_all_parameters(env_config, policy_config, {}, args)
    if task_config:
        ConfigLoader.merge_all_parameters(task_config, policy_config, {}, args)
    
    return policy_config, env_config, task_config, policy_cfg_path


def main(args):
    """Main training function."""
    args.is_training = True
    
    # Load configs
    policy_config, env_config, task_config, policy_cfg_path = load_configs(args)
    policy_type = policy_config.get('type', '')
    
    # Set seed
    if env_config:
        seed = int(env_config.get('seed', 0))
    elif task_config:
        seed = int(task_config.get('seed', 0))
    else:
        seed = 0
    set_seed(seed)
    logger.info(f"🌱 Seed: {seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save policy metadata
    metadata_path = os.path.join(args.output_dir, 'policy_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'policy_module': policy_type,
            'policy_name': policy_config.get('name', ''),
        }, f, indent=2)
    
    # Get trainer class dynamically
    trainer_class = get_policy_trainer_class(policy_cfg_path)
    if trainer_class is None:
        raise ValueError(f"No trainer_class found in policy config for {policy_type}")
    logger.info(f"Using trainer class: {trainer_class.__name__}")
    
    # Create environment/dataset and load agent based on policy type
    if policy_type.startswith('policy.rl.online.'):
        # Online RL: Create environment dynamically
        from policy.rl.base import RLConfig, RLMode
        import importlib
        
        env_type = env_config.get('type', '')
        if not env_type:
            raise ValueError("env config must have 'type' field (e.g., 'benchmark.dmc')")
        
        # Dynamically import env module
        if '.' in env_type:
            # Full path like 'benchmark.dmc'
            env_module = importlib.import_module(env_type)
            env_name = env_type.split('.')[-1]
        else:
            # Simple name like 'dmc'
            env_module = importlib.import_module(f"benchmark.{env_type}")
            env_name = env_type
        
        if not hasattr(env_module, 'create_env'):
            raise AttributeError(f"Env module {env_type} has no 'create_env' function")
        
        torch.backends.cudnn.benchmark = True
        
        # Extract parameters from env_config (needed for both vectorized and standard env creation)
        env_name_param = env_config.get('task', 'cartpole_swingup')
        image_size = int(env_config.get('image_size', 84))
        action_repeat = int(env_config.get('action_repeat', 4))
        frame_stack = int(env_config.get('frame_stack', 3))
        
        # Prepare env config (convert dict to namespace-like object for create_env)
        from types import SimpleNamespace
        env_cfg_obj = SimpleNamespace(**env_config)
        env_cfg_obj.seed = seed
        
        # Check if vectorized env creation is available
        num_envs = int(policy_config.get('num_envs', 1))
        make_vector_fn_name = f"make_vector_{env_name}_env"
        
        if num_envs > 1 and hasattr(env_module, make_vector_fn_name):
            # Use vectorized env creation if available
            make_vector_fn = getattr(env_module, make_vector_fn_name)
            
            env = make_vector_fn(
                env_name=env_name_param,
                num_envs=num_envs,
                seed=seed,
                image_size=image_size,
                action_repeat=action_repeat,
                frame_stack=frame_stack,
            )
        else:
            # Use standard create_env interface
            if num_envs > 1:
                # For vectorized envs without make_vector_* function, create multiple envs manually
                try:
                    from tianshou.env import SubprocVectorEnv
                except ImportError:
                    from benchmark.utils import SequentialVectorEnv as SubprocVectorEnv
                
                def make_env_fn(seed_offset):
                    def _make():
                        cfg = SimpleNamespace(**env_config)
                        cfg.seed = seed + seed_offset
                        return env_module.create_env(cfg)
                    return _make
                env_fns = [make_env_fn(i) for i in range(num_envs)]
                env = SubprocVectorEnv(env_fns)
            else:
                env = env_module.create_env(env_cfg_obj)
        
        # Create eval env (always single)
        eval_cfg_obj = SimpleNamespace(**env_config)
        eval_cfg_obj.seed = seed + 100
        eval_env = env_module.create_env(eval_cfg_obj)
        
        # Set env in args for load_model
        args.env = env
        
        # Load agent
        logger.info(f"Loading policy: {policy_cfg_path}")
        model_components = load_policy_model_for_training(policy_cfg_path, args, env_config)
        agent = model_components.get('agent') or model_components.get('model')
        if agent is None:
            raise ValueError(f"Failed to load agent. Expected 'agent' or 'model' key.")
        
        # Create RLConfig
        obs_shape = env.observation_space.shape
        max_episode_steps = getattr(env, '_max_episode_steps', 1000)
        
        rl_config = RLConfig(
            mode=RLMode.ONLINE,
            buffer_capacity=int(policy_config.get('replay_buffer_capacity', 100000)),
            buffer_device='cpu',
            batch_size=int(policy_config.get('batch_size', 128)),
            learning_rate=float(policy_config.get('lr', 1e-3)),
            discount=float(policy_config.get('discount', 0.99)),
            tau=float(policy_config.get('critic_tau', 0.01)),
            total_steps=int(policy_config.get('num_train_steps', 1000000)),
            eval_freq=int(policy_config.get('eval_freq', 10000)),
            save_freq=int(policy_config.get('save_freq', 50000)),
            log_freq=int(policy_config.get('log_freq', 1000)),
            warmup_steps=int(policy_config.get('num_seed_steps', 1000)),
            env_steps_per_train=int(policy_config.get('num_train_iters', 1)),
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            num_eval_episodes=int(policy_config.get('num_eval_episodes', 10)),
            output_dir=args.output_dir,
            seed=seed,
            device=str(policy_config.get('device', 'cuda')),
            ctrl_space='joint',
            ctrl_type='abs',
            chunk_size=1,
        )
        
        # Create trainer
        trainer = trainer_class(
            config=rl_config,
            policy=agent,
            env=env,
            eval_env=eval_env,
            image_pad=int(policy_config.get('image_pad', 4)),
            image_size=image_size,
        )
        trainer.save_video = bool(policy_config.get('save_video', True))
        
        # Cleanup
        def cleanup():
            if hasattr(env, 'close'):
                env.close()
            if hasattr(eval_env, 'close'):
                eval_env.close()
        
    elif policy_type.startswith('policy.rl.offline.'):
        # Offline RL: Load dataset
        from policy.rl.offline import load_d4rl_to_replay_buffer
        from policy.rl.base import RLConfig, RLMode
        
        if task_config.get('type') != 'benchmark.d4rl':
            raise ValueError(f"Offline RL requires task config with type=benchmark.d4rl, got {task_config.get('type')}")
        
        env_name = task_config.get('env', 'halfcheetah-medium-v2')
        logger.info(f"Loading D4RL dataset: {env_name}")
        
        eval_env, replay_buffer = load_d4rl_to_replay_buffer(
            env_name=env_name,
            device=str(policy_config.get('device', 'cuda')),
            storage_device='cpu',
        )
        
        # Set replay_buffer in args for load_model
        args.replay_buffer = replay_buffer
        
        # Load agent
        logger.info(f"Loading policy: {policy_cfg_path}")
        model_components = load_policy_model_for_training(policy_cfg_path, args, task_config)
        agent = model_components.get('agent') or model_components.get('model')
        if agent is None:
            raise ValueError(f"Failed to load agent. Expected 'agent' or 'model' key.")
        
        # Create RLConfig
        rl_config = RLConfig(
            mode=RLMode.OFFLINE,
            batch_size=int(policy_config.get('batch_size', 256)),
            learning_rate=float(policy_config.get('lr', 3e-4)),
            discount=float(policy_config.get('discount', 0.99)),
            total_steps=int(policy_config.get('num_steps', 1000000)),
            eval_freq=int(policy_config.get('eval_freq', 5000)),
            save_freq=int(policy_config.get('save_freq', 50000)),
            log_freq=int(policy_config.get('log_freq', 1000)),
            num_eval_episodes=int(policy_config.get('num_eval_episodes', 10)),
            max_episode_steps=int(task_config.get('max_episode_steps', 1000)),
            output_dir=args.output_dir,
            seed=seed,
            device=str(policy_config.get('device', 'cuda')),
        )
        
        # Create trainer
        trainer = trainer_class(
            config=rl_config,
            policy=agent,
            replay_buffer=replay_buffer,
            eval_env=eval_env,
        )
        
        # Cleanup
        def cleanup():
            if eval_env is not None and hasattr(eval_env, 'close'):
                eval_env.close()
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    
    # Resume from checkpoint if specified
    resume = policy_config.get('resume', None)
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)
    
    # Train
    logger.info(f"Starting {policy_type} training...")
    try:
        trainer.train()
    finally:
        cleanup()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    args = parse_param()
    main(args)
