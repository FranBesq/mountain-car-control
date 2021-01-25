import gym
import numpy as np
from simple_pid import PID

from env.mountain_car_env import Continuous_MountainCarEnv

env = Continuous_MountainCarEnv()
GOAL_POS = env.goal_position

# Enjoy best solution so far
k_p = -5 # 5
k_i = 0.40339942 # -5
k_d = -5 # -5

pid = PID(k_p, k_i, k_d, setpoint=1)

# Initilize sim
obs = env.reset()
done = False

while not done:
    env.render()
    err = min(obs[0] - GOAL_POS, 0)
    action = pid(err)
    obs, reward, done, info = env.step(action)

env.close()