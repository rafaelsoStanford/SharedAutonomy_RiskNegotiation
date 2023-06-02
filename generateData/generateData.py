'''
Using conda environement robodiff 3.9.15 from diffusion policy: https://diffusion-policy.cs.columbia.edu/
Follow their instructions to install the environment and dependencies
Copied their implmentation of ReplayBuffer into the Utils folder.
This ensures that the replay buffer is compatible with their environment when training diffusion in Colab file.
'''

import os
import shutil
import zarr
import numpy as np

from utils.replay_buffer import ReplayBuffer
from utils.car_racing import CarRacing
from utils.functions import *


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
    buffer = ReplayBuffer.create_empty_numpy()
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        obs_hist, act_hist = [], []
        obs = env.reset()
        done = False

        max_iter = 100
        iter = 0

        while not done:
            env.render(mode = "rgb_array")
            action = env.action_space.sample()
            
            # Save the observation and action            
            obs_hist.append(obs)
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
            
            if iter == max_iter:
                done = True
            iter += 1

        print("Episode finished after {} timesteps".format(len(obs_hist)))
            


def sinusoidalDriverSafe(env, TARGET_VELOCITY, NUM_EPISODES):
    pass

def sinusoidalDriverUnsafe(env, TARGET_VELOCITY, NUM_EPISODES):
    pass

def generateData():

    # Parameters for all three data gathering methods
    TARGET_VELOCITY = 30
    NUM_EPISODES = 10
    
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

