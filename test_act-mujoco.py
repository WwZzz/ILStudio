from benchmark.act_mujoco import ACTEnv
from method.act_policy import ACTAgent
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.robot.libero.libero_utils import (
    save_rollout_video,
)

DATE_TIME = time.strftime("%Y_%m_%d")

def eval_libero(task_suite_name='libero_spatial', num_trials_per_task = 50, local_log_dir: str = "./experiments/logs", ) -> None:
    run_id = f"EVAL-AGENT-{task_suite_name}-{DATE_TIME}"
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Initialize LIBERO task suite
    print(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    env = ACTEnv()
    task_description = env.get_task_prompt()
    agent = ACTAgent(
        ckpt_path='/mnt/afs/embodied_group8/wz/act-main/train_tmp/policy_best.ckpt',
        max_steps=env.get_max_steps()
    )
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")
        # Reset environment
        obs = env.reset(episode_idx=episode_idx)
        agent.reset()
        # Setup
        t = 0
        replay_images = []
        max_steps = env.get_max_steps()
        print(f"Starting episode {task_episodes+1}...")
        log_file.write(f"Starting episode {task_episodes+1}...\n")
        while t < max_steps:
            observation = agent.process_observation(obs)
            action = agent.act({'obs':observation, 'task_prompt': task_description, 't':t})
            res = env.step(action)
            obs, reward, done, info = res['obs'], res['reward'], res['done'], res['info']
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
        task_episodes += 1
        total_episodes += 1
        # Save a replay video of the episode
        save_rollout_video(
            replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
        )

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {total_episodes}")
        print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        log_file.write(f"Success: {done}\n")
        log_file.write(f"# episodes completed so far: {total_episodes}\n")
        log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
    # Save local log file
    log_file.close()


if __name__ == "__main__":
    eval_libero()


    # Initialize Weights & Biases logging as well
    # if cfg.use_wandb:
    #     wandb.init(
    #         entity=cfg.wandb_entity,
    #         project=cfg.wandb_project,
    #         name=run_id,
    #     )
    # Push total metrics and local log file to wandb
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             "success_rate/total": float(total_successes) / float(total_episodes),
    #             "num_episodes/total": total_episodes,
    #         }
    #     )
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
    #             f"num_episodes/{task_description}": task_episodes,
    #         }
    #     )
    #     wandb.save(local_log_filepath)