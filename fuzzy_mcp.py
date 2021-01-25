import gym
import matplotlib
from time import sleep
import numpy as np
# Fuzzy lib
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Import mountain env and initialize it
from env.mountain_car_env import Continuous_MountainCarEnv

MIN_POSITION = -1.2
MAX_POSITION = 0.6 

MIN_ACTION = -1
MAX_ACTION = 1

MAX_SPEED = 0.7
MIN_SPEED = -0.7

universe = np.linspace(0, 1, 5)

#Define Fuzzy control variables
position = ctrl.Antecedent(universe, 'position')  # min value, max value, output membership
velocity = ctrl.Antecedent(universe, 'velocity')      # min value, max value, output membership
output = ctrl.Consequent(universe, 'output')# min value, max value, output membership

# Generate membership functions
names = ['nb', 'ns', 'ze', 'ps', 'pb']
position.automf(names=names)
velocity.automf(names=names)
output.automf(names=names)

#Define ruleset

rule0 = ctrl.Rule(antecedent=((position['nb'] & velocity['nb']) |
                              (position['ns'] & velocity['nb']) |
                              (position['nb'] & velocity['ns'])),
                  consequent=output['nb'], label='rule nb')

rule1 = ctrl.Rule(antecedent=((position['nb'] & velocity['ze']) |
                              (position['nb'] & velocity['ps']) |
                              (position['ns'] & velocity['ns']) |
                              (position['ns'] & velocity['ze']) |
                              (position['ze'] & velocity['ns']) |
                              (position['ze'] & velocity['nb']) |
                              (position['ps'] & velocity['nb'])),
                  consequent=output['ns'], label='rule ns')

rule2 = ctrl.Rule(antecedent=((position['nb'] & velocity['pb']) |
                              (position['ns'] & velocity['ps']) |
                              (position['ze'] & velocity['ze']) |
                              (position['ps'] & velocity['ns']) |
                              (position['pb'] & velocity['nb'])),
                  consequent=output['ze'], label='rule ze')

rule3 = ctrl.Rule(antecedent=((position['ns'] & velocity['pb']) |
                              (position['ze'] & velocity['pb']) |
                              (position['ze'] & velocity['ps']) |
                              (position['ps'] & velocity['ps']) |
                              (position['ps'] & velocity['ze']) |
                              (position['pb'] & velocity['ze']) |
                              (position['pb'] & velocity['ns'])),
                  consequent=output['ps'], label='rule ps')

rule4 = ctrl.Rule(antecedent=((position['ps'] & velocity['pb']) |
                              (position['pb'] & velocity['pb']) |
                              (position['pb'] & velocity['ps'])),
                  consequent=output['pb'], label='rule pb')

system = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4])

# Later we intend to run this system with a 21*21 set of inputs, so we allow
# that many plus one unique runs before results are flushed.
# Subsequent runs would return in 1/8 the time!
sim = ctrl.ControlSystemSimulation(system)

# Initilize sim
env = Continuous_MountainCarEnv()
obs = env.reset()
GOAL_POS = env.goal_position
done = False

while not done:
    env.render()
    pos = (obs[0] + 1.2) / 1.8
    vel = ((obs[1]*10)+0.7)/ 1.4

    print("Posicion y velocidad: "+ str(pos) +" | " + str(vel)) #DEBUG
    
    # Set input to fuzzy control
    sim.input['position'] = pos
    sim.input['velocity'] = vel
    # Calculate consequent action
    sim.compute()
    action = sim.output['output']
    # Discretize action space
    if action < 0.4:
        action = MIN_ACTION
    else: 
        action = MAX_ACTION
    print("action is: " + str(action))#DEBUG

    obs, reward, done, info = env.step(action)

env.close()