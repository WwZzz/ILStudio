from benchmark.base import MetaEnv
import libero.libero.benchmark as libero_bench
import os
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

class LIBEROEnv(MetaEnv):
    MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object":280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    def __init__(self, task_suite_name:str='libero_spatial', task_id=0, resolution=256, seed=0, max_steps=-1, num_steps_wait=10):
        super(LIBEROEnv, self).__init__()
        assert task_suite_name in self.MAX_STEPS.keys()
        benchmark_dict = libero_bench.get_benchmark_dict()
        self.task_suite = benchmark_dict[task_suite_name]()
        self.task = self.task_suite.get_task(task_id)
        self.initial_states = self.task_suite.get_task_init_states(task_id)
        self.task_prompt = self.task.language
        self.num_steps_wait = num_steps_wait
        self.max_steps = max_steps if max_steps > 0 else self.MAX_STEPS[task_suite_name]
        self.env = OffScreenRenderEnv(**{
            "bddl_file_name": os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file),
            "camera_heights": resolution,
            "camera_widths": resolution
        })
        self.env.seed(seed)

    def reset(self, episode_idx):
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[episode_idx])
        if self.num_steps_wait>0:
            for _ in range(self.num_steps_wait):
                obs, reward, done, info = self.env.step([0, 0, 0, 0, 0, 0, -1])
        return obs

    def get_task_prompt(self):
        return self.task_prompt

    def step(self, action):
        obs, reward, done, info = self.env.step(action.tolist())
        return {'obs': obs, 'reward': reward, 'done': done, 'info': info}












