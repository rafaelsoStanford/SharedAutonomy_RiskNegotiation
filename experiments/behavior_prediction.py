import os
import shutil
import numpy as np

import time

import torch

from utils.car_racing import CarRacing
from utils.functions import *

# 1. Load the model
model = torch.load('experiments/model/Model_DifferentDriverBehaviors.pt')

# 2. Set up an environment
env = CarRacing()
env.render()
isopen = True
while isopen:
    start = time.time()
    total_reward = 0.0
    steps = 0
    restart = False
    env.reset()
    while True:
        if time.time() - start < 1:
            print("Waiting for zoom to end")
            a = np.array([0.0, 0.0, 0.0])
            s, r, done, info = env.step(a)
            continue

        a = env.action_space.sample()
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        isopen = env.render()
        if done or restart or isopen == False:
            break
env.close()
