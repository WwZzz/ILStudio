import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4")
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

