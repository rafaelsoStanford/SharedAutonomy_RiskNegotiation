import numpy as np
import time
import torch

from utils.car_racing import CarRacing
from utils.functions import *
from utils.experiment_funcs import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from modules.denoisingNet import *
import pickle

import matplotlib.pyplot as plt



# 1. Set up the environment and parameters
model = torch.load('experiments/models/DifferentDrivingBehaviors_medium.pt')

env = CarRacing()
env.render()

pred_horizon = 8    # Number of steps we predict into the future
obs_horizon = 3     # Number of steps we condition on
action_horizon = 2

vision_feature_dim = 512    # ResNet18 has output dim of 512
lowdim_obs_dim = 1 + 3      # Car Velocity is 1 dimensional
obs_dim = vision_feature_dim + lowdim_obs_dim    # observation feature has 514 dims in total per step
action_dim = 3 + 1         # Action space is 3 dimensional + 1 for track_flag

num_diffusion_iters = 100   # Number of diffusion iterations

device = torch.device('cuda')
ema_net = CreateDenoisingNet(
    pred_horizon,
    obs_horizon,
    action_horizon,
    vision_feature_dim,
    lowdim_obs_dim,
    obs_dim,
    action_dim,
    num_diffusion_iters,
    device=device
)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',    # the choice of beta schedule has a big impact on performance
    clip_sample=True,    # clip output to [-1,1] to improve stability
    prediction_type='epsilon'    # our network predicts noise (instead of denoised action)
)

ema_net.load_state_dict(model)
print('Pretrained weights loaded.')

# 2. Load pickle file with stats from dataset model was trained on
with open('experiments/file.pkl', 'rb') as f:
    stats = pickle.load(f)

# 3. Experimental Parameters
target_velocity = 20    # Target velocity of the car
env.seed(7000)    # Set a seed for fixed starting conditions (same environment)
start_iter = 300    # We shall move away from the starting position for 300 steps using PD controller
isopen = True

# Experimental runs
# We want to do two experimental runs: One with diffusion prediction / one with "human" inputs
diffusion_output = {}
controller_output = {}

for run in range(2):
    obs = env.reset() # Reset the environment
    done = False
    start = time.time() # Start timer to circumvent zooming in at the beginning
    steps = 0
    while not done:
        if time.time() - start < 1: 
            # We wait for 1 second
            a = np.array([0.0, 0.0, 0.0])
            s, r, done, info = env.step(a) # Step has to be called for the environment to continue
            continue

        #  Set up an observation dictionary
        observation = { 
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
        
        # Reached Starting position
        if run == 0:
            ''' EXPERIMENT 1: Diffusion Policy Prediction:
                We gather T_obs steps of history and predict the next T_pred steps.

                Output: Action sequence of length T_pred
                        Track flag sequence of length T_pred
            '''
            # Condition on the obs_horizon steps
            img_hist, vel_hist, haction_hist = gatherConditionalActions(env, observation, obs_horizon, start_iter)
            # Make predictions
            noutput = Inference(
                ema_net=ema_net,
                img_hist=img_hist,
                vel_hist=vel_hist,
                haction_hist=haction_hist,
                pred_horizon=pred_horizon,
                num_diffusion_iters=num_diffusion_iters,
                action_dim=action_dim,
                noise_scheduler=noise_scheduler,
                stats=stats,
                device=device
            )

            #Split actions (dim 3) and track (dim 1):
            naction = noutput[:, 0:3]
            ntrack = noutput[:,-1]

            action_pred = unnormalize_data(naction, stats=stats['action'])
            track_pred = unnormalize_data(ntrack, stats=stats['track'])

            diffusion_output['actions'] = action_pred # dim (T_pred, 3)
            diffusion_output['track'] = track_pred # dim (T_pred, 1)

            done = True

        if run == 1:
            ''' EXPERIMENT 2: Controller Policy Prediction:
                We gather T_pred steps that would actually be done by "human"

                Output: Action sequence of length T_pred
                        Track flag sequence of length T_pred
            '''
            a_hist, t_hist = [] , []
            for _ in range(pred_horizon):
                # EXPERIMENT 2: "Human" (ie controller) inputs
                # We move using the "human" policy
                a = action_sinusoidalTrajectory(steps, 1/100, observation, 12 , target_velocity) #Unsafe driver
                obs, r, done, info = env.step(a)
                steps += 1

                observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation 
                    "image": obs,
                    "velocity": env.return_absolute_velocity(),     # This function was added manually to car racing environment
                    "track": env.return_track_flag()                # This function was added manually to car racing environment
                }
                # We store the history
                a_hist.append(a)
                t_hist.append(observation["track"])

            controller_output['actions'] = a_hist # dim (T_pred, 3)
            controller_output['track'] = t_hist # dim (T_pred, 1)

            done = True


print("Diffusion Output: ", diffusion_output['actions'])
print("Controller Output: ", controller_output['actions'])

print("Diffusion Output: ", diffusion_output['track'].astype(int))
print("Controller Output: ", controller_output['track'])

# 4. Plotting
# We plot the track flag and the actions along T_pred steps
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.plot(diffusion_output['track'].astype(int), label='Diffusion')
plt.plot(controller_output['track'], label='Controller')
plt.xlabel('Time Steps')
plt.ylabel('Track Flag')
plt.legend()

# Change all elements of controller to array elements to match diffusion
controller_output['actions'] = np.array(controller_output['actions'])

plt.subplot(1, 4, 2)
plt.plot(diffusion_output['actions'][:, 0], label='Diffusion')
plt.plot(controller_output['actions'][:, 0], label='Controller')
plt.xlabel('Time Steps')
plt.ylabel('Steering')

plt.subplot(1, 4, 3)
plt.plot(diffusion_output['actions'][:, 1], label='Diffusion')
plt.plot(controller_output['actions'][:, 1], label='Controller')
plt.xlabel('Time Steps')
plt.ylabel('Acceleration')

plt.subplot(1, 4, 4)
plt.plot(diffusion_output['actions'][:, 2], label='Diffusion')
plt.plot(controller_output['actions'][:, 2], label='Controller')
plt.xlabel('Time Steps')
plt.ylabel('Brake')

plt.legend()
plt.show()



