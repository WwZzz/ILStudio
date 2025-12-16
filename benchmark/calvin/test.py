import numpy as np 

from calvin_env_10.envs.play_table_env import get_env 

task = "task_D" # available tasks: task_A, task_B, task_C

env = get_env(task,show_gui=False)

start_obs = env.reset()

print(start_obs)

done = False
while not done:

  # Relative actions
  action = np.array([0,0,0,0,0,0,1], dtype=np.float32)
  
  # Absolute actions
  #action = (
  #      np.array([0, 0, 0], dtype=np.float32),
  #      np.array([0, 0, 0], dtype=np.float32),
  #      np.array([1], dtype=np.float32),
  #  )
    
  
  obs, reward, done, info = env.step(action)

  print(obs)
  print(reward)
  print(done)
  print(info)

  print("--------------------------------")
  break
