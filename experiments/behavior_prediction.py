import os
import shutil
import numpy as np

import time

import torch

from utils.car_racing import CarRacing
from utils.functions import *

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from modules.denoisingNet import *


# 1. Load the model
model = torch.load('experiments/models/Model_DifferentDriverBehaviors.pt')

# 2. Set up an environment
env = CarRacing()
env.render()

# Parameters
pred_horizon = 8
obs_horizon = 3
action_horizon = 2

# ResNet18 has output dim of 512
vision_feature_dim = 512
# Car Velocity is 1 dimensional
lowdim_obs_dim = 1
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
# Action space is 3 dimensional
action_dim = 3

ema_net = CreateDenoisingNet(pred_horizon, obs_horizon, action_horizon,
                             vision_feature_dim, lowdim_obs_dim, obs_dim, action_dim,
                             num_diffusion_iters=100)

#state_dict = torch.load(ckpt_path, map_location='cuda')
ema_net.load_state_dict(model)
print('Pretrained weights loaded.')


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
