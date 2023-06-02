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


def keyboardControl():
    pass

if __name__ == "__main__":
    # Define dictionary of functions to switch between based on method_id
    method_id = "slalom"
    switch_dict = {
        "default": keyboardControl
    }

    # Call the appropriate function based on method_id
    switch_case = switch_dict.get(method_id, switch_dict["default"])
    switch_case()

