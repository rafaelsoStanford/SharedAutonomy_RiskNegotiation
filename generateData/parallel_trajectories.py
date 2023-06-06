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

    return mask_yellow, mask_blue, mask_cyan, mask_magenta, mask_purple
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

def run(env):
    # Simply run a trajectory for a bit to get the environment started
    env.reset()
    isopen = True
    while isopen:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        image = obs.copy()

        # Render all trajectories using masks:
        mask_yellow, mask_blue, mask_cyan, mask_magenta, mask_purple = maskTrajecories(image)

        # Display the masks:
        # cv2.imshow("yellow", mask_yellow)
        # cv2.imshow("blue", mask_blue)
        # cv2.imshow("cyan", mask_cyan)
        # cv2.imshow("magenta", mask_magenta)
        # cv2.imshow("purple", mask_purple)


        posB2vec = env.return_carPosition()
        # print("x position: ", posB2vec.x)
        # print("y position: ", posB2vec.y)

        pixelpos = (int(posB2vec.x), int(posB2vec.y))
        print("pixel position: ", pixelpos)


        

        isopen = env.render()



def generateData():
    # Init environment and buffer
    env = CarRacing()
    env.render(mode="human")

    run(env) # Run the environment for a bit to get it started



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

