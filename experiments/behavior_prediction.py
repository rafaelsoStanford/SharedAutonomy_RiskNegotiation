import os
import shutil
import numpy as np

import time

import torch

from utils.car_racing import CarRacing
from utils.functions import *


# 1. Load the model
model = torch.load('experiments/models/Model_DifferentDriverBehaviors.pt')

# 2. Set up an environment
env = CarRacing()
env.render()

# 3. Experimental Parameters:
target_velocity = 20 # Target velocity of the car
env.seed(7000) # Set a seed for fixed starting conditions (same environment)
start_iter = 300 # We shall move away from the starting position for 300 steps using PD controller



isopen = True

while isopen:
    start = time.time()
    total_reward = 0.0
    steps = 0
    restart = False
    obs = env.reset()
    while True:

        if time.time() - start < 1:
            print("Waiting for zoom to end")
            a = np.array([0.0, 0.0, 0.0])
            s, r, done, info = env.step(a) # Step has to be called for the environment to move forward
            continue

        observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
            "image": obs,
            "velocity": env.return_absolute_velocity(), # This function was added manually to car racing environment
            "track": env.return_track_flag()
        }

        if steps < start_iter:
            # Use PD controller to move away from starting position
            a = calculateAction(observation, target_velocity)
        else: 
            a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        isopen = env.render()
        if done or restart or isopen == False:
            break
env.close()
