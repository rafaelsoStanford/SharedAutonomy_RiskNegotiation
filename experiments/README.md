### Experiments ###

# Car Racing Diffusion Model

This code implements a car racing diffusion model using a previously trained model and controllers mimicing human driving behaviors. It performs two experimental runs: one with diffusion policy prediction and another with controller policy prediction. The code uses the Gym library version 0.21.0.
Finally prediction and actual controller are compared to each other. 

## Python Version for Gym
The code is compatible with Gym library version 0.21.0. Make sure to install this specific version to ensure compatibility.

## Setup

Download the pretrained model and the `stats` pickle file (run the colab file and save said files locally).
   - Place the pretrained model file (e.g., `DifferentDrivingBehaviors.pt`) in the `experiments/models` folder.
   - Save the `stats` pickle file in the `experiments` folder. This file contains the observed min-max values of the trained dataset,
       which are used to unnormalize the data and regain the former signal sizes.
       
       
The code will perform two experimental runs: one with diffusion policy prediction and another with controller policy prediction. The predicted actions and track flags will be stored in the output dictionaries (`diffusion_output` and `controller_output`).

## Important Functions

**Environment related functions / Agent related
- `calculateAction`: This function calculates the action using a PD controller to move away from the starting position based on the current observation and target velocity.

- `action_sinusoidalTrajectory`: This function generates actions using a sinusoidal trajectory controller for the controller policy prediction experiment based on the current number of steps, frequency, observation, duration, and target velocity.

- `gatherConditionalActions`: This function gathers the conditional actions (image history, velocity history, and historical actions) for the diffusion prediction experiment based on the specified observation horizon and start iteration.

**Diffusion policy related functions (See colab implementation)

- `CreateDenoisingNet`: This function sets up the denoising network model with the specified parameters for the prediction horizon, observation horizon, action horizon, and dimensions of the observation and action spaces.

- `DDPMScheduler`: This class sets up the scheduler for the diffusion model training with the specified parameters, including the number of diffusion iterations, beta schedule, clipping, and prediction type.

- `Inference`: This function performs the inference using the denoising network model, given the gathered conditional actions, prediction horizon, number of diffusion iterations, action dimension, noise scheduler, statistics, and device.









