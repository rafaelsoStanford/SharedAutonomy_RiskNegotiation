'''
Using conda environement gym_0.21.0
Copied  implmentation of ReplayBuffer from diffusion policy  into the Utils folder.
This ensures that the replay buffer is compatible with their environment when training diffusion in Colab file.

IMPORTANT:  The current environment of car_race is not compatible with gym or gymnasium (which would be version 0.26.0).
            It is compatible with gym==0.21.0!!! 
            This is because the car_racing.py file was modified and doing it again for the new version of gym would be a pain.
            Maybe I will do it in the future, but for now, I will stick with gym==0.21.0
'''

import os
import shutil
import zarr
import numpy as np

import time

from utils.replay_buffer import ReplayBuffer
from utils.car_racing import CarRacing
from utils.functions import *


def action_sinusoidalTrajectory(t, freq, observation, Amplitude, target_velocity):
    # Observations are the following:
    image = observation['image']
    velocity = observation['velocity']

    # Environment constants
    carPos = np.array([70, 48]) # Position of the car in the image (pixel coordinates)
    widthOfTrack = 20 # Approx width of the track in pixels

    # Initialize controllers
    pid_angle = PID(0.5, -0.2, 0.0, setpoint=0)
    pid_velocity = PID(0.05, 0.1, 0.1, setpoint=target_velocity)

    # Find the next target point of sinusoidal trajectory
    scale_dist = 10 # This scales the vertical distance of the next target point from tip of car
    targetPoint, estimatedMiddlePoint, vector_track_normalized, vector_track_perp_normalized = calculateTargetPoint(image, widthOfTrack, freq, scale_dist , Amplitude, t)
    
    if targetPoint is None:
        action = [0,0,0] # If unreasonable values where found for the target point, keep the previous action. This avoids an edge case error
        return action

    # Calculate the angle to the target point
    error = targetPoint - carPos
    carVector = np.array([-1, 0])
    angle = np.arccos(np.dot(error, carVector) / (np.linalg.norm(error) * np.linalg.norm(carVector)))
    #Check if the angle is positive or negative -> negative full left turn, positive full right turn        
    if error[1] > 0:
        angle = -angle        
    steeringAngle = pid_angle(angle)
    # Calculate the acceleration or if negative, the breaking
    acc = pid_velocity(velocity)
    breaking = 0
    if acc < 0:
        breaking = -acc
        acc = 0
    action = [steeringAngle, acc, breaking]

    #print("Actions: ", action)
    return action
    

def keyboardControl():

    # This is the code from the car_racing.py file
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])
    val = 1.0
    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -val
        if k == key.RIGHT:
            a[0] = +val
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -val:
            a[0] = 0
        if k == key.RIGHT and a[0] == +val:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False    
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
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

def pidDriver(env, TARGET_VELOCITY, NUM_EPISODES):
    
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        img_hist, vel_hist ,act_hist, flag_hist = [], [], [], []
        obs = env.reset()
        done = False

        max_iter = 1000
        iter = 0

        while not done:
            env.render(mode="rgb_array")

            observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
                "image": obs,
                "velocity": env.return_absolute_velocity(), # This function was added manually to car racing environment
                "track": env.return_track_flag()
            }

            action = calculateAction(observation, TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(observation['image'])
            vel_hist.append(observation['velocity'])
            flag_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1

        print("Episode finished after {} timesteps".format(len(img_hist)))
        
def sinusoidalDriverSafe(env, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)

        # Parameters for sinusoidal trajectory
        Amplitude = 5 #Safe (ie within bounds of track); found by trial and error
        freq = 1/100 

        img_hist, vel_hist ,act_hist, flag_hist = [], [], [], []
        obs = env.reset()
        done = False

        max_iter = 1000
        iter = 0

        while not done:
            env.render(mode="rgb_array")

            observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
                "image": obs,
                "velocity": env.return_absolute_velocity(), # This function was added manually to car racing environment
                "track": env.return_track_flag()
            }

            action = action_sinusoidalTrajectory(iter, freq, observation, Amplitude ,TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(observation['image'])
            vel_hist.append(observation['velocity'])
            flag_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1

        print("Episode finished after {} timesteps".format(len(img_hist)))

def sinusoidalDriverUnsafe(env, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)

        # Parameters for sinusoidal trajectory
        Amplitude = 13 #Safe (ie within bounds of track); found by trial and error
        freq = 1/100 

        img_hist, vel_hist ,act_hist, flag_hist = [], [], [], []
        obs = env.reset()
        done = False

        max_iter = 1000
        iter = 0

        while not done:
            env.render(mode="rgb_array")

            observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
                "image": obs,
                "velocity": env.return_absolute_velocity(), # This function was added manually to car racing environment
                "track": env.return_track_flag()
            }

            action = action_sinusoidalTrajectory(iter, freq, observation, Amplitude ,TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(observation['image'])
            vel_hist.append(observation['velocity'])
            flag_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1

        print("Episode finished after {} timesteps".format(len(img_hist)))

def generateData():
    # Parameters for all three data gathering methods
    TARGET_VELOCITY = 30
    NUM_EPISODES = 1
    
    # Create the environment
    env = CarRacing()
    env.render(mode="rgb_array")

    pidDriver(env, TARGET_VELOCITY, NUM_EPISODES)
    sinusoidalDriverSafe(env, TARGET_VELOCITY, NUM_EPISODES)
    sinusoidalDriverUnsafe(env, TARGET_VELOCITY, NUM_EPISODES)

if __name__ == "__main__":
    ''' USAGE: 
        adjust method_id to select the method of data generation

            default:        Allows for keyboard control of the car. No data is saved. Meant as a way to test the environment.
            generateData:   Generates three types of data: PD-Driver, Sinusoidal-Driver(safe) and Sinusoidal-Driver(unsafe). 
                            Saves data to the specified directory.
    '''

    method_id = "generateData"
    switch_dict = {
        "default": keyboardControl,
        "generateData": generateData
    }

    # Call the appropriate function based on method_id
    switch_case = switch_dict.get(method_id, switch_dict["default"])
    switch_case()

