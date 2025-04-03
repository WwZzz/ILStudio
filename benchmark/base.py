import numpy as np

class MetaEnv:
    def __init__(self):
        self.max_steps = np.inf

    def get_max_steps(self):
        return self.max_steps

    def get_task_prompt(self):
        return ''

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError



