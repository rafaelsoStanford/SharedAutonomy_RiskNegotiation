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

import pickle


with open('experiments/file.pkl', 'rb') as f:
    stats = pickle.load(f)


# 1. Load the model
model = torch.load('experiments/models/DifferentDrivingBehaviors.pt')

# 2. Set up an environment
env = CarRacing()
env.render()

# Parameters
pred_horizon = 8    # Number of steps we predict into the future
obs_horizon = 3     # Number of steps we condition on
action_horizon = 2

# ResNet18 has output dim of 512
vision_feature_dim = 512
# Car Velocity is 1 dimensional
lowdim_obs_dim = 1 + 3
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
# Action space is 3 dimensional
action_dim = 3 + 1
num_diffusion_iters=100

device = torch.device('cuda')
ema_net = CreateDenoisingNet(pred_horizon, obs_horizon, action_horizon,
                             vision_feature_dim, lowdim_obs_dim, obs_dim, action_dim,
                             num_diffusion_iters, device=device)

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
    img_hist, vel_hist, haction_hist = [], [], [] # Initialize history
    while True:

        if time.time() - start < 1: 
            print("Waiting for zoom to end")
            a = np.array([0.0, 0.0, 0.0])
            s, r, done, info = env.step(a) # Step has to be called for the environment to move forward
            continue

        observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
            "image": obs,
            "velocity": env.return_absolute_velocity(),     # This function was added manually to car racing environment
            "track": env.return_track_flag()                # This function was added manually to car racing environment
        }

        if steps < start_iter:
            # Use PD controller to move away from starting position
            a = calculateAction(observation, target_velocity)
            obs, r, done, info = env.step(a)
            steps += 1
            continue

        # Condition on the obs_horizon steps

        for i in range(obs_horizon):
            # We move using the PD controller
            a = calculateAction(observation, target_velocity)
            # We calculate the "human" policy ie sinonoidal trajectory
            h_action = calculateAction(observation, target_velocity) #action_sinusoidalTrajectory(i+start_iter, 1/100, observation, 5, target_velocity) #Unsafe driver
            obs, r, done, info = env.step(a)

            observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
                "image": obs,
                "velocity": env.return_absolute_velocity(),     # This function was added manually to car racing environment
                "track": env.return_track_flag()                # This function was added manually to car racing environment
            }
            # We store the history
            img_hist.append(observation["image"])
            vel_hist.append(observation["velocity"])
            haction_hist.append(h_action)

        img_hist = np.array(img_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        vel_hist = vel_hist[:,None]
        haction_hist = np.array(haction_hist, dtype=np.float32)

############################################ 
        # stack the last obs_horizon number of observations
        images = np.stack(img_hist)
        images = np.moveaxis(images, -1,1) # normalize
        velocity =  np.stack(vel_hist)
        h_action = np.stack(haction_hist)
        # normalize observation
        #stats = get_data_stats(velocity)
        nvelocity = normalize_data(velocity,stats = stats['velocity'])
        #stats = get_data_stats(h_action)
        nh_action = normalize_data(velocity, stats=stats['h_action'])
        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
        # (2,3,96,96)
        nvelocity = torch.from_numpy(nvelocity).to(device, dtype=torch.float32)
        #nvelocity = nvelocity.unsqueeze(-1)
        nh_action = torch.from_numpy(nh_action).to(device, dtype=torch.float32)
        #nh_action = nh_action.unsqueeze(-1)

        # infer action
        with torch.no_grad():
            # get image features
            image_features = ema_net['vision_encoder'](nimages)
            # (2,512)
            # concat with low-dim observations
            obs_features = torch.cat([image_features, nvelocity, nh_action], dim=-1)
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (1, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_net['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        noutput = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        noutput = noutput[0] # Since batch size is 1

        #Split actions (dim 3) and track (dim 1):
        naction = noutput[:, 0:3]
        ntrack = noutput[:,-1]

        action_pred = unnormalize_data(naction, stats=stats['action'])
        track_pred = unnormalize_data(ntrack, stats=stats['track'])

        print("Action: ", action_pred)
        print("Track: ", track_pred)

        # # only take action_horizon number of actions
        # start = obs_horizon - 1
        # end = start + action_horizon
        # action = action_pred[start:end,:]

####################

        steps += 1
        isopen = env.render()
        if done or restart or isopen == False:
            break
env.close()
