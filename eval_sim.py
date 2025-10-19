import os
from tianshou.env import SubprocVectorEnv
import json
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
    parser.add_argument('--is_pretrained', action='store_true', default=True,
                       help='Whether to use pretrained model')
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
    parser.add_argument('--num_rollout', type=int, default=4,
                       help='Number of rollouts')
    parser.add_argument('--num_envs', type=int, default=2,
                       help='Number of environments')
    parser.add_argument('--use_spawn', action='store_true',
                       help='Use spawn method for multiprocessing')
    # Model parameters (will be loaded from checkpoint config if not provided)
    parser.add_argument('--chunk_size', type=int, default=64,
                       help='Actual chunk size for policy that will truncate each raw chunk')
    # Parse arguments
    args, unknown = parser.parse_known_args()
    # keep unknown tokens for env overrides (e.g., --env.task)
    args._unknown = unknown
    return args

if __name__=='__main__':
    set_seed(0)
    args = parse_param()
    if args.use_spawn: mp.set_start_method('spawn', force=True)
    policy = load_policy(args)

    # load env via YAML config
    from configs.loader import ConfigLoader
    cfg_loader = ConfigLoader(args=args, unknown_args=getattr(args, '_unknown', []))
    env_cfg, env_cfg_path = cfg_loader.load_env(args.env)
    # sync derived values from env config
    if hasattr(env_cfg, 'task'):
        args.task = env_cfg.task
    if hasattr(env_cfg, 'max_timesteps'):
        args.max_timesteps = env_cfg.max_timesteps
    env_module = importlib.import_module(f"benchmark.{env_cfg.type}")
    if not hasattr(env_module, 'create_env'): raise AttributeError(f"env {env_cfg.type} has no 'create_env'")
    def env_fn(env_config, env_handler):
        def create_env():
            return env_handler(env_config)
        return create_env

    all_eval_results = []
    num_iters = args.num_rollout//args.num_envs if args.num_rollout%args.num_envs==0 else args.num_rollout//args.num_envs+1
    for i in tqdm(range(args.num_rollout//args.num_envs), total=num_iters):
        num_envs = args.num_envs if i<num_iters-1 else args.num_rollout-i*args.num_envs
        # init video recorder
        if args.output_dir!='':
            video_dir = os.path.join(args.output_dir, env_cfg.type, 'video')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{args.task}_roll{i*args.num_envs}_{i*args.num_envs+num_envs}.mp4") 
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
        print(eval_result)
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
    # save result
    if args.output_dir!='':
        env_res_dir = os.path.join(args.output_dir, env_cfg.type)
        os.makedirs(env_res_dir, exist_ok=True)
        env_res_file = os.path.join(env_res_dir, f'{args.task}.json')
        # eval_result = {k:v.astype(np.float32) if isinstance(v, np.ndarray) else v for k,v in eval_result.items()}
        with open(env_res_file, 'w') as f:
            json.dump(eval_result, f)

    