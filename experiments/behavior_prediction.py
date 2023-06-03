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

from modules.UNet import *
from modules.getResnet import *


# 1. Load the model
model = torch.load('experiments/models/Model_DifferentDriverBehaviors.pt')

# 2. Set up an environment
env = CarRacing()
env.render()

# Parameters
pred_horizon = 8
obs_horizon = 3
action_horizon = 2

#Set up model for evaluation
#@markdown ### **Network Demo**

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder = get_resnet('resnet18')

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_bn_with_gn(vision_encoder)


## Dimensions of Features ## 

# ResNet18 has output dim of 512
vision_feature_dim = 512
# Car Velocity is 1 dimensional
lowdim_obs_dim = 1
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
# Action space is 3 dimensional
action_dim = 3

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

# demo
with torch.no_grad():
    # example inputs
    image = torch.zeros((1, obs_horizon,3,96,96))
    agent_vel = torch.zeros((1, obs_horizon, 1))
    # vision encoder
    image_features = nets['vision_encoder'](
        image.flatten(end_dim=1))
    # (2,512)
    image_features = image_features.reshape(*image.shape[:2],-1)
    # (1,2,512)
    obs = torch.cat([image_features, agent_vel],dim=-1)
    # (1,2,514)

    noised_action = torch.randn((1, pred_horizon, action_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = nets['noise_pred_net'](
        sample=noised_action, 
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise 
    # the actual noise removal is performed by NoiseScheduler 
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)


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
