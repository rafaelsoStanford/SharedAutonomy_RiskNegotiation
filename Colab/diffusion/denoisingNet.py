
import torch
import torch.nn as nn
import numpy as np
from utils.functions import *

from Unet import *
from getResnet import *

def CreateDenoisingNet(pred_horizon, obs_horizon, action_horizon, vision_feature_dim, lowdim_obs_dim, 
                       obs_dim, action_dim, num_diffusion_iters, device):
    #Set up model for evaluation
    #@markdown ### **Network Demo**

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    _ = nets.to(device)

    return nets


def Inference(  ema_net,
                img_hist,
                vel_hist,
                haction_hist,
                pred_horizon,
                num_diffusion_iters,
                action_dim,
                noise_scheduler,
                stats,
                device):
    
    """
    Perform prediction of future actions using the trained model.
    """
    


    # stack the last obs_horizon number of observations
    images = np.stack(img_hist)
    images = np.moveaxis(images, -1,1) # normalize
    velocity =  np.stack(vel_hist)
    h_action = np.stack(haction_hist)
    # normalize observation
    #stats = get_data_stats(velocity)
    nvelocity = normalize_data(velocity,stats = stats['velocity'])
    #stats = get_data_stats(h_action)
    nh_action = normalize_data(h_action, stats=stats['h_action'])
    # images are already normalized to [0,1]
    nimages = images

    # device transfer
    nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
    # (2,3,96,96)
    nvelocity = torch.from_numpy(nvelocity).to(device, dtype=torch.float32)
    nh_action = torch.from_numpy(nh_action).to(device, dtype=torch.float32)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = ema_net['vision_encoder'](nimages)
        # (2,512)
        # concat with low-dim observations
        obs_features = torch.cat([image_features, nvelocity, nh_action], dim=-1)
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (1, pred_horizon, action_dim), device=device)
        naction = noisy_action
        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_net['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

    # unnormalize action
    noutput = naction.detach().to('cpu').numpy()
    # (B, pred_horizon, action_dim)
    noutput = noutput[0] # Since batch size is 1
    return noutput