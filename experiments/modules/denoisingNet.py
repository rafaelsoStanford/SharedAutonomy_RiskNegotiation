
from modules.UNet import *
from modules.getResnet import *

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

    # demo
    with torch.no_grad():
        # example inputs
        image = torch.zeros((1, obs_horizon,3,96,96))
        agent_vel = torch.zeros((1, obs_horizon, 1))
        # vision encoder
        image_features = nets['vision_encoder'](
            image.flatten(end_dim=1))
        # (2,512)
        image_features = image_features.reshape(*image.shape[:2],-1)
        # (1,2,512)
        obs = torch.cat([image_features, agent_vel],dim=-1)
        # (1,2,514)

        noised_action = torch.randn((1, pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = nets['noise_pred_net'](
            sample=noised_action, 
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))

        # illustration of removing noise 
        # the actual noise removal is performed by NoiseScheduler 
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    _ = nets.to(device)

    return nets