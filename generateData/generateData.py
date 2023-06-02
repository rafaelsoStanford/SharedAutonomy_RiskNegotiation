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

import time

from utils.replay_buffer import ReplayBuffer
from utils.car_racing import CarRacing
from utils.functions import *
from utils.controller import Controller # Original controller implementation by Rafael Sonderegger, opted using simplePid library instead
from simple_pid import PID


def find_edge_1dStrip(array, direction):
    # Find edge point of a 1D array. 
    # If none is found return -1
    starting_point = int(len(array) // 2)
    idx = -1
    if direction == 'left':
        for i in range(starting_point, -1, -1):
            if array[i] != 0:
                idx = i
                break
    elif direction == 'right':
        for i in range(starting_point, len(array)):
            if array[i] != 0:
                idx = i
                break
    return idx

def find_middle_point(strip_1d):
    # Check if there is edge point for both left and right side of track.
    # If none is found set border of strip as edge point
    idx1 = find_edge_1dStrip(strip_1d, 'left')
    idx2 = find_edge_1dStrip(strip_1d, 'right')

    if idx1 == -1:
        idx1 = 0
    if idx2 == -1:
        idx2 = len(strip_1d) - 1

    idx_middle = int((idx1 + idx2) / 2)
    return idx_middle

def calculateDistAngle(idx_middle_upper, idx_middle_lower, strip_width, strip_height):
    # Calculate distance and angle from middle of the track
    # idx_middle_upper: index of middle point on upper edge of strip
    # idx_middle_lower: index of middle point on lower edge of strip
    # strip_width: width of the strip
    # strip_height: height of the strip
    # return: distance and angle
    
    # Compute distance to middleline
    distance_to_middleline = strip_width // 2 - idx_middle_lower
    # Compute angular error
    upper_lenght_to_target = strip_width // 2 - idx_middle_upper
    angle_to_target = np.arctan(upper_lenght_to_target / strip_height)
    return distance_to_middleline, angle_to_target


def processImage(image):
    # Cropping image down to a strip
    strip_height = 20
    strip_width = 96
    middle_height = 65
    top = int(middle_height - strip_height / 2)
    bottom = int(middle_height + strip_height / 2)
    # Crop the strip from the image
    strip = image[top:bottom, :]

    ## Mask where only edge is retained
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    imask_green = mask_green>0
    gray_mask = imask_green.astype(np.uint8)
    gray_mask = gray_mask*255
    # Only use two edges of the strip: Upper and lower and find edge points coordinates
    upper_edge = gray_mask[0, :]
    lower_edge = gray_mask[strip_height - 1, :]
    # Get index of middle point on the upper and lower edge
    idx_middle_upper = find_middle_point(upper_edge)
    idx_middle_lower = find_middle_point(lower_edge)

    distance, angle = calculateDistAngle(idx_middle_upper, idx_middle_lower, strip_width, strip_height)
    return distance, angle


def calculateAction(observation , target_velocity):

    # Initialize controllers
    pid_angle = PID(0.5, -0.01, 0.05, setpoint=0)
    pid_distance = PID(0.5, -0.005, 0.05, setpoint=0)
    pid_velocity = PID(0.05, -0.1, 0.2, setpoint=target_velocity)
    

    # Distinguish observation type
    image = observation['image']
    velocity = observation['velocity']

    # Get distance from processed image
    error_dist, error_ang = processImage(image)

    # Get control outputs from PD controllers
    control_ang = pid_angle(error_ang)
    control_dist = pid_distance(error_dist)
    control_vel = pid_velocity(velocity)

    print("Control outputs: ", control_ang, control_dist, control_vel)
    acc = control_vel
    breaking = 0
    if acc < 0:
        acc = 0
        breaking = -control_vel
    
    # Calculate and return final action
    action = (control_ang , acc, breaking)
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
    controller = Controller()
    buffer = ReplayBuffer.create_empty_numpy()
    for episode in range(NUM_EPISODES):
        print("Episode: ", episode)
        img_hist, vel_hist ,act_hist = [], [], []
        obs = env.reset()
        done = False

        max_iter = 1000
        iter = 0

        while not done:
            env.render(mode = "human")

            observation = { #In order to be more consistent, we will group state variables used for training in a dictionary called observation
                "image": obs,
                "velocity": env.return_absolute_velocity() # This function was added manually to car racing environment
            }

            action = calculateAction(observation, TARGET_VELOCITY)
            
            # Save the observation and action            
            img_hist.append(observation['image'])
            vel_hist.append(observation['velocity'])
            act_hist.append(action)

            print("Velocity: ", observation['velocity'])
            # Take the action
            obs, reward, done, info = env.step(action)
        
            if iter == max_iter:
                done = True
            iter += 1

        print("Episode finished after {} timesteps".format(len(img_hist)))
            


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

