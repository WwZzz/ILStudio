import hydra
from hydra.utils import instantiate
from benchmark.base import *
import calvin_env

class CalvinEnv(MetaEnv):
    """
    calvin环境
    """
    def __init__(self, config_path, config_name, overrides=None):
        """
        初始化并加载底层环境。

        Args:
            config_path (str): calvin_env 'conf' 目录的相对路径。
            config_name (str): 主配置文件名。
            overrides (list, optional): 用于覆盖配置的字符串列表。
        """
        print("SimpleCalvinWrapper: 正在加载环境...")
        if overrides is None:
            overrides = []

        with hydra.initialize(config_path=config_path, job_name="simple_calvin_wrapper"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            # 将实例化的环境存储在公有属性 self.env 中
            env = instantiate(cfg.env)
        super().__init__(env)

    def reset(self):
        """
        重置环境。
        
        Returns:
            obs: 环境的初始观测。
        """
        return self.env.reset()

    def step(self, action):
        """
        在环境中执行一个动作。

        Returns:
            tuple: 一个四元组 (obs, reward, done, info)。
        """
        return self.env.step(action)

    def render(self):
        """
        渲染一帧环境画面（如果 GUI 已启用）。
        """
        return self.env.render()

    def close(self):
        """
        关闭环境并释放资源。
        """
        print("SimpleCalvinWrapper: 正在关闭环境...")
        self.env.close()