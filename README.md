# SharedAutonomy_RiskNegotiation

Shared autonomy for transitions from AI to human control of a vehicle. Design of an agent capable identifying human driver behavior and safely transitioning vehicle control.

## Features

- **Behavior Identification**: The agent is equipped with algorithms and models to identify various human driver behaviors, allowing it to make informed decisions during control transitions.

- **Control Transition**: The system is designed to seamlessly transfer control of the vehicle from the AI agent to the human driver. The transition should be smooth after Risk has been evaluated.

- **Risk Negotiation**: The agent employs risk negotiation strategies to assess potential risks associated with control transitions and make proactive decisions to mitigate them. This ensures that the vehicle operates in a safe and reliable manner.

- **Diffusion Model Training**: The repository also includes a data generation script that can be used to train diffusion models. These models can be used to model and predict the behavior of human drivers in distinct scenarios.
-> Currently uses a Google Colab implementation which is based on the code of the following work: [Diffusion Policy
Visuomotor Policy Learning via Action Diffusion] https://github.com/columbia-ai-robotics/diffusion_policy 

## Prerequisites

```
...

```

## Usage

- **Data Generation**: In the corresponding folder you will find data generation script which is based on the CarRacing environment by OpenAi Gym. It uses a slightly modified version of `car_racing.py`in order to account for the desired observations. The data is saved using a .zarr structure and consequently as a.zip file, such that it is compatible with the implementation in Google Colab.
- **Diffusion (Colab)**: 
The Colab implementation is based on the image colab repository made by [Diffusion Policy
Visuomotor Policy Learning via Action Diffusion] https://github.com/columbia-ai-robotics/diffusion_policy . We replaced the environment to use the CarDriving-v2 from OpenAi **Gymnasium** (which is equivalent to my knowledge to gym==0.26.0). Since Colab switched to Python version 3.10.x the gym==0.21.0 is no longer compatible. Thus we modified the CarRacing environment of Gymnasium. Thus note that there are syntax differences between how `env.` functions are used, compared to our data generation files. 
The environment inputs and outputs have stayed the same, thus the data can still be used. 
