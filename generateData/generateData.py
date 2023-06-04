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

def pidDriver(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        img_hist, vel_hist ,act_hist, track_hist = [], [], [], []
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
            track_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1

        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        track_hist = np.array(track_hist, dtype=np.float32)

        episode_data = {"img": img_hist, 
                        "velocity": vel_hist[:, None], 
                        "track": track_hist[:, None],
                        "action": act_hist, 
                        "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                        }
        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()
    return img_hist, vel_hist ,act_hist, track_hist

        
def sinusoidalDriverSafe(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)

        # Parameters for sinusoidal trajectory
        Amplitude = 5 #Safe (ie within bounds of track); found by trial and error
        freq = 1/100 

        img_hist, vel_hist ,act_hist, track_hist = [], [], [], []
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
            track_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1
        
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        track_hist = np.array(track_hist, dtype=np.float32)

        episode_data = {"img": img_hist, 
                "velocity": vel_hist[:, None], 
                "track": track_hist[:, None],
                "action": act_hist, 
                "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                }
        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()
    return img_hist, vel_hist ,act_hist, track_hist

def sinusoidalDriverUnsafe(env, buffer, TARGET_VELOCITY, NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)

        # Parameters for sinusoidal trajectory
        Amplitude = 13 #Safe (ie within bounds of track); found by trial and error
        freq = 1/100 

        #Initialize the buffers
        img_hist, vel_hist ,act_hist, track_hist = [], [], [], []
        
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
            track_hist.append(observation['track'])
            act_hist.append(action)

            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1
        
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        track_hist = np.array(track_hist, dtype=np.float32)

        episode_data = {"img": img_hist, 
                "velocity": vel_hist[:, None], 
                "track": track_hist[:, None],
                "action": act_hist, 
                "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                }
        buffer.add_episode(episode_data)

        print("Episode finished after {} timesteps".format(len(img_hist)))
        env.close()
    return img_hist, vel_hist ,act_hist, track_hist

def generateData():
    # Parameters for all three data gathering methods
    TARGET_VELOCITY = 30
    NUM_EPISODES = 10
    CHUNK_LEN = -1

    #Path to save data
    path = "./data/multipleDrivingBehaviours.zarr"
    
    # Init environment and buffer
    env = CarRacing()
    env.render(mode="rgb_array")
    buffer = ReplayBuffer.create_empty_numpy()

    # Run the data gathering methods and save in history lists
    pidDriver(env, buffer,TARGET_VELOCITY, NUM_EPISODES)
    sinusoidalDriverSafe(env , buffer, TARGET_VELOCITY, NUM_EPISODES)
    sinusoidalDriverUnsafe(env, buffer, TARGET_VELOCITY, NUM_EPISODES)

    print("Saving data to path: ", path)
    buffer.save_to_path(path, chunk_length=CHUNK_LEN)

    # Consolidate metadata and zip the file
    store = zarr.DirectoryStore(path)
    data = zarr.group(store=store)
    print(data.tree(expand=True))
    zarr_file = os.path.basename(path)
    zip_file = zarr_file + ".zip"
    zarr.consolidate_metadata(store)
    shutil.make_archive(path, "zip", path)
    print(f"Zarr file saved as {zip_file}")
    


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

