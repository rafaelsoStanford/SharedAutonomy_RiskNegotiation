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
import gym
import math

import time
from simple_pid import PID

from utils.replay_buffer import ReplayBuffer
from utils.car_racing import CarRacing
from utils.functions import *

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


def findClosestPoint(trajectory_img, carPos = np.array([70, 48])):
    trajectory_idx = np.nonzero(trajectory_img)
    #Find single closest edge point
    distanceCarToTraj = np.linalg.norm(np.array(carPos)[:, None] - np.array(trajectory_idx), axis=0)
    closetPointIdx = np.argmin(distanceCarToTraj)
    closestPoint = np.array([trajectory_idx[0][closetPointIdx], trajectory_idx[1][closetPointIdx]])
    return closestPoint

def run(env, agent: int):
    # Simply run a trajectory for a bit to get the environment started
    env.reset()
    isopen = True
    target_velocity = 40      
    action = np.array([0, 0, 0], dtype=np.float32)
    pid_velocity = PID(1.0, 0.05 , 0.2, setpoint=target_velocity, output_limits=(-1, 1))
    pid_steering = PID(0.5, 0.01, 0.05, setpoint=0)
    #pid_steering = PID(0.9, 0.06, 0.4 , setpoint=0,  differential_on_measurement=False, proportional_on_measurement=False)
    pid_steering.output_limits = (-1, 1)
    car_pos_vector = np.array([70, 48]) # Car remains fixed relativ to window 
    obs, reward, done, info = env.step(action) # Take a step to get the environment started (action is empty)
    
    while isopen:       
        isopen = env.render()
        augmImg = info['augmented_img']
        velB2vec = info['car_velocity_vector']
        posB2vec = info['car_position_vector']

        out = augmImg.copy()
        
        v = np.linalg.norm(velB2vec)
        # Render all trajectories using masks:
        dict_masks = maskTrajecories(augmImg)
        track_img = dict_masks[agent] # Get the correct mask for the desired agent
        # Get single line strip in front of car
        line_strip = track_img[60, :]
        idx = np.nonzero(line_strip)[0]

        if len(idx) == 0: # Rarely happens, but sometimes the is no intersection of trajectory with line strip -> continue with previous action
            obs, reward, done, info = env.step(action)
            continue 

        # Get index closest to middle of strip (idx = 48)
        idx = idx[np.argmin(np.abs(idx - 48))]
        target_point = np.array([60, idx])
        car2point_vector = target_point - car_pos_vector
        

        # Draw line from car to target point
        # Draw target point
        # Draw mask

        cv2.circle(out, (target_point[1], target_point[0]), 2, (255, 0, 0), -1)
        cv2.line(out, (car_pos_vector[1], car_pos_vector[0]), (car_pos_vector[1] + car2point_vector[1], car_pos_vector[0] + car2point_vector[0]), (0, 0, 255), 2)
        cv2.imshow('Trajectory', track_img)
        cv2.imshow('AugmImage', out)
        cv2.imshow('image', obs)
        cv2.waitKey(0)  

        # As an approximation let angle be the x component of the car2point vector
        err =  idx - 48 #-car2point_vector[1] # Correcting for the fact that negative value is a left turn, ie positive angle
        err = np.clip(err, -5, 5)

        angle = np.arctan2(abs(err), abs(car2point_vector[0]))
        if err > 0:
            angle = -angle
        print("PID Steering:", pid_steering(angle))
        action[0] = pid_steering(angle)

        print("Angle: ", angle)
        print("Error: ", err)
        print("Action: ", action[0])
        print("Speed: ", v)
        if pid_velocity(v) < 0:
            action[1] = 0
            action[2] = pid_velocity(v)
        else:
            action[1] = pid_velocity(v)
            action[2] = 0

        obs, reward, done, info = env.step(action)




def generateData():
    # Init environment and buffer
    env = CarRacing()
    env.render(mode="human")
    run(env, 'left')

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

