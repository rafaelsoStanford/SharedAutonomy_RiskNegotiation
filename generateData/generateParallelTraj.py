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

import matplotlib.pyplot as plt
import random


def findClosestPoint(trajectory_img, carPos = np.array([70, 48])):
    trajectory_idx = np.nonzero(trajectory_img)
    #Find single closest edge point
    distanceCarToTraj = np.linalg.norm(np.array(carPos)[:, None] - np.array(trajectory_idx), axis=0)
    closetPointIdx = np.argmin(distanceCarToTraj)
    closestPoint = np.array([trajectory_idx[0][closetPointIdx], trajectory_idx[1][closetPointIdx]])
    return closestPoint


def maskTrajecories(image):
    # Define the threshold ranges for each mask
    lower_yellow = np.array([100, 100, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 0], dtype=np.uint8)

    lower_blue = np.array([0, 0, 100], dtype=np.uint8)
    upper_blue = np.array([0, 0, 255], dtype=np.uint8)

    lower_cyan = np.array([0, 100, 100], dtype=np.uint8)
    upper_cyan = np.array([0, 255, 255], dtype=np.uint8)

    lower_magenta = np.array([100, 0, 100], dtype=np.uint8)
    upper_magenta = np.array([255, 0, 255], dtype=np.uint8)

    lower_purple = np.array([100, 0 , 100], dtype=np.uint8)
    upper_purple = np.array([200, 50,200], dtype=np.uint8)

    # Apply the masks
    mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(image, lower_blue, upper_blue)
    mask_cyan = cv2.inRange(image, lower_cyan, upper_cyan)
    mask_magenta = cv2.inRange(image, lower_magenta, upper_magenta)
    mask_purple = cv2.inRange(image, lower_purple, upper_purple)

    # Label the differently colored trajectories
    dict_masks = {'lleft': mask_yellow, 
                  'left': mask_cyan, 
                  'middle': mask_magenta, 
                  'right': mask_purple, 
                  'rright': mask_blue}

    return dict_masks

def switch_mode(modes):
    chosen_mode = random.choice(modes)
    return chosen_mode



def driving(env, buffer , TARGET_VELOCITY = 10, NUM_EPISODES = 10, MODE = 'middle'):
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        #Initialize the buffers
        img_hist, vel_hist ,act_hist, pos_hist = [], [], [], []
        
        #Init Params
        action = np.array([0, 0, 0], dtype=np.float32)
        pid_velocity = PID(0.01, 0, 0.05, setpoint=float(TARGET_VELOCITY), output_limits=(0, 1))
        pid_breaking = PID(0.05, 0.00, 0.08, setpoint=float(TARGET_VELOCITY), output_limits=(0, 0.9))
        
        pid_steering = PID(0.8, 0.01, 0.3, setpoint=0, output_limits=(-1, 1))
        
        car_pos_vector = np.array([70, 48]) # Car remains fixed relativ to window 

        #Init Env
        isopen = True
        env.reset()
        obs, reward, done, info = env.step(action) # Take a step to get the environment started (action is empty)
        absVel = []
        max_steps = 1000
        for i in range(max_steps):
            isopen = env.render("state_pixels")
            augmImg = info['augmented_img']
            velB2vec = info['car_velocity_vector']
            posB2vec = info['car_position_vector']  

            carVelocity_wFrame = [velB2vec.x , velB2vec.y]
            carPosition_wFrame = [posB2vec.x , posB2vec.y]

            if i % 200 == 0: # All 10 steps, we change velocity
                velocitites = [10 ,20 ,30]
                TARGET_VELOCITY = random.choice(velocitites)
                pid_velocity.setpoint = TARGET_VELOCITY
                # print("Target Velocity: ", TARGET_VELOCITY)
            
            v = np.linalg.norm(velB2vec)
            absVel.append(v)
            # Render all trajectories using masks:
            dict_masks = maskTrajecories(augmImg)
            track_img = dict_masks[MODE] # Get the correct mask for the desired agent
            # Get single line strip in front of car
            line_strip = track_img[60, :]
            idx = np.nonzero(line_strip)[0]

            if len(idx) == 0: # Rarely happens, but sometimes the is no intersection of trajectory with line strip -> continue with previous action
                action[0] = -0.6
                obs, reward, done, info = env.step(action)
                continue

            # Get index closest to middle of strip (idx = 48)
            idx = idx[np.argmin(np.abs(idx - 48))]
            target_point = np.array([60, idx])
            car2point_vector = target_point - car_pos_vector# As an approximation let angle be the x component of the car2point vector
            
            # As an approximation let angle be the x component of the car2point vector
            err =  idx - 48 #-car2point_vector[1] # Correcting for the fact that negative value is a left turn, ie positive angle
            err = np.clip(err, -5, 5)

            angle = np.arctan2(abs(err), abs(car2point_vector[0]))
            if err > 0:
                angle = -angle
            # print("PID Steering:", pid_steering(angle))
            action[0] = pid_steering(angle)

            if abs(action[0]) > 0.8:
                action[2] = 0.9
                

            # print("Angle: ", angle)
            # print("Error: ", err)
            # print("Action: ", action[0])
            # print("Speed: ", v)

            if TARGET_VELOCITY - v < -5:
                action[1] = 0
                action[2] = pid_breaking(v)
                #action[2] = pid_velocity(v)
                #action[2] = np.clip(action[2], 0, 0.8)
            else:
                action[1] = pid_velocity(v)
                action[2] = 0

            obs, reward, done, info = env.step(action)
            #print(action)
            # Save the observation and action            
            img_hist.append(obs)
            vel_hist.append(carVelocity_wFrame)
            pos_hist.append(carPosition_wFrame)
            act_hist.append(action)

            # print("Angle: ", angle)
            # print("Error: ", err)
            # print("Action: ", action[0])
            # print("Speed: ", v)

            # out = augmImg.copy()
            # cv2.circle(out, ([idx, 60]), 2, (255, 0, 0), -1)
            
            # cv2.imshow('Trajectory', track_img)
            # cv2.imshow('AugmImage', out)
            # cv2.imshow('image', obs)
            # cv2.waitKey(1)  

            # Take the action
            obs, reward, done, info = env.step(action)

            
        
            if done or isopen == False:
                break
        
        img_hist = np.array(img_hist, dtype=np.float32)
        act_hist = np.array(act_hist, dtype=np.float32)
        vel_hist = np.array(vel_hist, dtype=np.float32)
        pos_hist = np.array(pos_hist, dtype=np.float32)

        episode_data = {"img": img_hist, 
                "velocity": vel_hist, 
                "position": pos_hist,
                "action": act_hist, 
                "h_action": act_hist #This will act as a placeholder for "human action". It is crude, but works for current testing purposes
                }
        buffer.add_episode(episode_data)

        # # Plot the absolute velocity history
        # plt.plot(absVel)
        # plt.show()
        # plt.waitforbuttonpress()

        print("Episode finished after {} timesteps".format(len(img_hist)))
    env.close()
    return img_hist, vel_hist ,act_hist, pos_hist

def generateData():
    # Parameters for all three data gathering methods
    TARGET_VELOCITY = 30
    NUM_EPISODES = 1
    CHUNK_LEN = -1

    #Path to save data
    path = "./data/multipleDrivingBehaviours.zarr"
    
    # Init environment and buffer
    env = CarRacing()
    env.render(mode="rgb_array")
    buffer = ReplayBuffer.create_empty_numpy()

    modes = ['lleft', 'left', 'middle', 'right', 'rright']
    
    for mode in modes:
        print("Mode: ", mode)
        img_hist, vel_hist ,act_hist, track_hist = driving(env, buffer, TARGET_VELOCITY, NUM_EPISODES, mode)

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
            generateData:   Generates 5 different datasets, each with a different driving behaviour, driving on different positions of the track
    '''
    generateData()
