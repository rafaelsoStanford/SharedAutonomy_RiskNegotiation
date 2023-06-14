import os
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl

from diffusion import *
from LoadCarRacingData import *
from utils.car_racing import *

import matplotlib.pyplot as plt

def main(n_epochs=100, AMP=True, batch_size=16):

    # Parameters:
    T_obs= 2
    T_pred=8
    T_act =1

    # Dimensions:
    # ResNet18 has output dim of 512
    vision_feature_dim = 512 #Costant

    # Position (2 dim) + Velocity (2 dim) + "Human" action (3 dim)
    lowdim_obs_dim = 2 + 2 + 3
    # observation feature total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim

    # Action space is 3 dimensional + Position 2 dim + Velocity 2 dim
    action_dim = 3 + 2 + 2
    
    # # ===========model===========
    dataset = LoadCarRacingData(data_dir="./data/multipleDrivingBehaviours_testing.zarr.zip", batch_size=1,
                                T_obs=T_obs, T_pred=T_pred , T_act =T_act)
    dataset.setup()
    dataloader = dataset.train_dataloader()
    
    diffusion = Diffusion(T_obs=T_obs, T_pred=T_pred , T_action =T_act, global_cond_dim=obs_dim ,diffusion_out_dim= action_dim)
    diffusion.load_checkpoint("model_parallel.ckpt")
    diffusion.eval()


    trainer = pl.Trainer()
    trainer.predict(model=diffusion, dataloaders=dataloader)






    


    # Sample from the model
    env = CarRacing()
    env.reset()
    
    # max_steps = 100
    # #Init Env
    # isopen = True
    # action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # obs, reward, done, info = env.step(action) # Take a step to get the environment started (action is empty)
    # absVel = []
    # carVel_hist, carPos_hist = [], []




    # for i in range(max_steps):
    #     # isopen = env.render("human")

    #     # augmImg = info['augmented_img']
    #     # velB2vec = info['car_velocity_vector']
    #     # posB2vec = info['car_position_vector']  

    #     # carVelocity_wFrame = [velB2vec.x , velB2vec.y]
    #     # carPosition_wFrame = [posB2vec.x , posB2vec.y]

    #     # carVel_hist.append(carVelocity_wFrame)
    #     # carPos_hist.append(carPosition_wFrame)
    #     # action = env.action_space.sample()

    #     # env.step(action)
    
    # Plot position in 2D space plane
    carVel_hist = np.array(carVel_hist)
    carPos_hist = np.array(carPos_hist)

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Modify the transformation of the axes
    ax.invert_xaxis()  # Invert the x-axis
    ax.invert_yaxis()  # Invert the y-axis

    # Plot every tenth position point
    x = [pos[0] for pos in carPos_hist[::10]]
    y = [pos[1] for pos in carPos_hist[::10]]
    ax.plot(x, y, 'o')

        # Scale factor for the vectors
    scaling_factor = 0.1

    # Calculate rescaled velocity vectors
    u = [vel[0] * scaling_factor for vel in carVel_hist[::10]]
    v = [vel[1] * scaling_factor for vel in carVel_hist[::10]]
    
    # Set color for the first point and vector
    color = 'green'

    # Plot the first point and vector in green
    ax.plot(x[0], y[0], 'o', color=color)
    ax.quiver(x[0], y[0], u[0], v[0], angles='xy', scale_units='xy', scale=1, color=color)

    # Plot the remaining points and vectors in default color
    ax.quiver(x[1:], y[1:], u[1:], v[1:], angles='xy', scale_units='xy', scale=1)

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Car Position and Velocity Vectors')

    # Show the plot
    plt.show()



if __name__ == "__main__":
    main()