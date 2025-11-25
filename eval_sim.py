import configs  # Must be first to suppress TensorFlow logs
import os
from tianshou.env import SubprocVectorEnv
import json
from loguru import logger
from data_utils.utils import set_seed
from tqdm import tqdm
import imageio
from benchmark.utils import evaluate
import importlib
import multiprocessing as mp
from policy.utils import load_policy

def parse_param():
    """
    Parse command line arguments using simple argparse.
    
    Returns:
        args: Parsed arguments namespace
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a policy model')
    parser.add_argument('-o', '--output_dir', type=str, default='results/dp_aloha_transer-official-ema-freq50-dnoise10-aug',
                    help='Directory to save results')
    # Model arguments
    parser.add_argument('-s', '--seed', type=int, default=0,
                       help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    # Model loading - can be checkpoint path or server address
    parser.add_argument('-m', '--model_name_or_path', type=str, 
                       default='localhost:5000',
                       help='Path to model checkpoint OR server address (host:port) for remote policy server')
    parser.add_argument('--dataset_id', type=str, default='',
                       help='Dataset ID to use (only for local model loading, ignored for remote server)')
    # Simulator arguments
    parser.add_argument('-e', '--env', type=str, default='aloha',
                       help='Env config (name under configs/env or absolute path to yaml)')
    # task/max_timesteps come from env config; override via --env.task / --env.max_timesteps if needed
    parser.add_argument('--fps', type=int, default=50,
                       help='Frames per second')
    parser.add_argument('-n', '--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('-bs', '--batch_size', type=int, default=2,
                       help='Number of parallel environments (batch size)')
    parser.add_argument('--use_spawn', action='store_true',
                       help='Use spawn method for multiprocessing')
    # Model parameters (will be loaded from checkpoint config if not provided)
    parser.add_argument('-ck', '--chunk_size', type=int, default=-1,
                       help='Actual chunk size for policy that will truncate each raw chunk')
    # Parse arguments
    args, unknown = parser.parse_known_args()
    # keep unknown tokens for env overrides (e.g., --env.task)
    args._unknown = unknown
    return args

if __name__=='__main__':
    args = parse_param()
    args.is_training = False
    set_seed(args.seed)
    if args.use_spawn: mp.set_start_method('spawn', force=True)
    policy = load_policy(args)

    # load env via YAML config
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, '_unknown', []))
    env_cfg, env_cfg_path = cfg_loader.load_env(args.env)
    
    # Support multiple environments in one config
    # Check if env_cfg is a list or has an 'envs' attribute for multiple environments
    if isinstance(env_cfg, list):
        env_cfgs = env_cfg
    else:
        env_cfgs = [env_cfg]
    
    def env_fn(env_config, env_handler):
        def create_env():
            return env_handler(env_config)
        return create_env
    
    # Store results for all environments
    all_env_results = {}
    
    # Iterate through each environment configuration
    for env_idx, env_cfg in enumerate(env_cfgs):
        logger.info("="*60)
        logger.info(f"Evaluating environment {env_idx + 1}/{len(env_cfgs)}")
        logger.info("="*60)
        
        # sync derived values from env config
        if hasattr(env_cfg, 'task'):
            args.task = env_cfg.task
        if hasattr(env_cfg, 'max_timesteps'):
            args.max_timesteps = env_cfg.max_timesteps
        
        # Parse env type - support both old format (simple name) and new format (full path)
        env_type = env_cfg.type
        if '.' in env_type:
            # New format: full path like 'benchmark.aloha.AlohaSimEnv'
            # Extract module path and class name
            module_path, class_name = env_type.rsplit('.', 1)
            env_module = importlib.import_module(module_path)
            # Use the module name (e.g., 'aloha') for directory naming
            env_name = module_path.split('.')[-1] if '.' in module_path else module_path
        else:
            # Old format: simple name like 'aloha'
            env_module = importlib.import_module(f"benchmark.{env_type}")
            env_name = env_type
        
        if not hasattr(env_module, 'create_env'): 
            raise AttributeError(f"env module {module_path if '.' in env_type else env_type} has no 'create_env'")
        
        all_eval_results = []
        num_iters = args.num_rollout//args.batch_size if args.num_rollout%args.batch_size==0 else args.num_rollout//args.batch_size+1
        for i in tqdm(range(args.num_rollout//args.batch_size), total=num_iters, desc=f"Env {env_idx+1} Rollouts"):
            num_envs = args.batch_size if i<num_iters-1 else args.num_rollout-i*args.batch_size
            # init video recorder
            if args.output_dir!='':
                video_dir = os.path.join(args.output_dir, env_name, 'video')
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"{args.task}_roll{i*args.batch_size}_{i*args.batch_size+num_envs}.mp4") 
                video_writer = imageio.get_writer(video_path, fps=args.fps)
            else:
                video_writer = None
            env_fns = [env_fn(env_cfg, env_module.create_env) for _ in range(num_envs)]
            env = SubprocVectorEnv(env_fns)
            # evaluate
            if hasattr(policy, 'policy') and hasattr(policy.policy, 'eval'):
                # Local model mode
                policy.policy.eval()
            # Remote mode doesn't need model.eval()
            
            eval_result = evaluate(args, policy, env, video_writer=video_writer)
            logger.info(eval_result)
            all_eval_results.append(eval_result)
            policy.reset()
        
        eval_result = {
            'total_success': sum(eri['total_success'] for eri in all_eval_results),
            'total': sum(eri['total'] for eri in all_eval_results),
            'horizon': sum([eri['horizon'] for eri in all_eval_results], []),
            'horizon_success': sum([eri['horizon_success']*eri['total_success'] for eri in all_eval_results])
        }
        eval_result['success_rate'] = 1.0*eval_result['total_success']/eval_result['total']    
        eval_result['horizon_success']/=eval_result['total_success']
        
        # Store results for this environment
        env_key = f"{env_name}_{args.task}" if hasattr(env_cfg, 'task') else f"{env_name}_env{env_idx}"
        all_env_results[env_key] = eval_result
        
        # save result
        if args.output_dir!='':
            env_res_dir = os.path.join(args.output_dir, env_name)
            os.makedirs(env_res_dir, exist_ok=True)
            env_res_file = os.path.join(env_res_dir, f'{args.task}.json')
            # eval_result = {k:v.astype(np.float32) if isinstance(v, np.ndarray) else v for k,v in eval_result.items()}
            with open(env_res_file, 'w') as f:
                json.dump(eval_result, f)
        
        logger.info(f"Environment {env_idx + 1} Results:")
        logger.info(f"  Success Rate: {eval_result['success_rate']:.2%}")
        logger.info(f"  Total Success: {eval_result['total_success']}/{eval_result['total']}")
        logger.info(f"  Avg Horizon (Success): {eval_result['horizon_success']:.2f}")
    
    # Print summary for all environments
    if len(env_cfgs) > 1:
        logger.info("="*60)
        logger.info("Summary of All Environments:")
        logger.info("="*60)
        for env_key, result in all_env_results.items():
            logger.info(f"{env_key}: Success Rate = {result['success_rate']:.2%} ({result['total_success']}/{result['total']})")
        
        # Calculate average success rate across all environments
        total_success_all = sum(result['total_success'] for result in all_env_results.values())
        total_all = sum(result['total'] for result in all_env_results.values())
        avg_success_rate_weighted = total_success_all / total_all if total_all > 0 else 0.0
        
        # Calculate simple average (unweighted)
        avg_success_rate_simple = sum(result['success_rate'] for result in all_env_results.values()) / len(all_env_results)
        
        # Calculate average horizon for successful episodes across all environments
        total_horizon_success_weighted = sum(
            result['horizon_success'] * result['total_success'] 
            for result in all_env_results.values()
        )
        avg_horizon_success = total_horizon_success_weighted / total_success_all if total_success_all > 0 else 0.0
        
        logger.info("="*60)
        logger.info("Overall Statistics:")
        logger.info(f"  Average Success Rate (Weighted): {avg_success_rate_weighted:.2%} ({total_success_all}/{total_all})")
        logger.info(f"  Average Success Rate (Simple): {avg_success_rate_simple:.2%}")
        logger.info(f"  Average Horizon (Success): {avg_horizon_success:.2f}")
        logger.info("="*60)
        
        # Prepare summary with overall statistics
        summary_data = {
            'individual_environments': all_env_results,
            'overall_statistics': {
                'avg_success_rate_weighted': avg_success_rate_weighted,
                'avg_success_rate_simple': avg_success_rate_simple,
                'total_success': total_success_all,
                'total': total_all,
                'avg_horizon_success': avg_horizon_success,
                'num_environments': len(all_env_results)
            }
        }
        
        # Save combined results
        if args.output_dir!='':
            summary_file = os.path.join(args.output_dir, 'all_envs_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
