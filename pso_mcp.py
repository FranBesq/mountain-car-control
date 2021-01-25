import gym
import numpy as np
import pso
from simple_pid import PID

from env.mountain_car_env import Continuous_MountainCarEnv

MAX_STEPS = 2000

def mountain_sim(x):
    # Get pid params from particle
    k_p = x[0]
    k_i = x[1]
    k_d = x[2]

    # Create the PID
    pid = PID(k_p, k_i, k_d, setpoint=1)

    # Initilize sim
    obs = env.reset()
    done = False
    reward = 0
    #err = 1/ (reward + 1)
    errTotal = 0 # Cumulative error is our cost 

    for i in range(MAX_STEPS):
        #err = abs(min(obs[0] - GOAL_POS, 0)) # absolute value of: Position - goal
        action = pid(reward)
        obs, reward, done, info = env.step(action)
        #err = 1/ (reward + 1)
        # Get cost/err
        errTotal += reward

        if done:
            break

    return errTotal


# Define Optimization Problem
problem = {
    'CostFunction': mountain_sim,
    'nVar': 3,
    'VarMin': -5,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
    'VarMax': 5,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
}

env = Continuous_MountainCarEnv()
GOAL_POS = env.goal_position

# Running PSO
pso.tic()
print('Running PSO ...')
gbest, pop = pso.PSO(problem, MaxIter=200, PopSize=25)
env.close()
pso.toc()
print('Global Best:')
print(gbest)
print()

# Enjoy best
k_p = gbest["position"][0]
k_i = gbest["position"][1]
k_d = gbest["position"][2]

pid = PID(k_p, k_i, k_d, setpoint=1)

# Initilize sim
env = Continuous_MountainCarEnv()
obs = env.reset()
done = False

while not done:
    env.render()
    err = min(obs[0] - GOAL_POS, 0)
    action = pid(err)
    obs, reward, done, info = env.step(action)

env.close()

