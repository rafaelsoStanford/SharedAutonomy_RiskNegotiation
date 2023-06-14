import torch
import torch.nn as nn
import pytorch_lightning as pl

from Unet import *
from denoisingNet import *

from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


from PIL import Image
import io
import numpy as np

class Diffusion(pl.LightningModule):
    def __init__(self, noise_steps=100
                , beta_start=1e-4
                , beta_end=0.02
                , img_size=94
                , img_channels=3
                , lr=1e-4
                , diffusion_out_dim = 3+2+2
                , global_cond_dim = 512 + 2+2+3
                , T_obs = 3
                , T_pred = 3
                , T_action = 1):
        
        super().__init__()
        ### Define parameters
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_action = T_action

        self.diffusion_out_dim = diffusion_out_dim
        self.global_cond_dim = global_cond_dim
        self.noise_steps = noise_steps

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.lr = lr
        self.loss = nn.MSELoss()

        ### Define model which will be a simplifed 1D UNet
        self.vision_encoder = VisionEncoder() # Loads pretrained weights of Resnet18 with output dim 512
        self.vision_encoder.device = self.device
        
        self.noise_predictor = ConditionalUnet1D(
                                    input_dim= diffusion_out_dim,
                                    global_cond_dim= global_cond_dim * T_obs)
        
        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_predictor': self.noise_predictor
        })

        print("Device of noise_predictor: ", next(self.noise_predictor.parameters()).device)
        print("Device of vision_encoder: ", next(self.vision_encoder.parameters()).device)
        print(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_steps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample = True,
            prediction_type = 'epsilon')
    
        self.ema = EMAModel(
                            parameters=self.nets,
                            power=0.75)
    
        

    def onepass(self, batch, batch_idx, mode):
        # Prepare data
        nimage = batch['image'][:,:self.T_obs]
        nVelocity_obs = batch['velocity_obs'][:,:self.T_obs]
        nPosition_obs = batch['position_obs'][:,:self.T_obs]
        nAction_obs = batch['action_obs'][:,:self.T_obs]
        B = nVelocity_obs.shape[0]

        # encoder vision features
        image_features = self.nets['vision_encoder'](
            nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nVelocity_obs, nPosition_obs, nAction_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)
        
        # Output: Get actions and future positions and velocities
        nAction = batch['actions_pred']
        nVelocity = batch['velocities_pred']
        nPosition = batch['positions_pred']

        # Concatenate actions and future positions and velocities
        actions_to_pred = torch.cat([nAction, nVelocity, nPosition], dim=-1)

        # sample noise to add to actions
        noise = torch.randn(actions_to_pred.shape, device=self.device)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (B,), device=self.device
        ).long()
        # Get noisy action using scheduler (forward process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions_to_pred, noise, timesteps)
        
        predicted_noise = self.nets['noise_predictor'](noisy_actions, timesteps, obs_cond) # Use the UNET to predict noise

        loss = self.loss(noise, predicted_noise)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="Train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="Val")
        self.log("val_loss",loss)

    def predict_step(self, batch, batch_idx):
        nimage_obs = batch['image'][:,:self.T_obs]
        nVelocity_obs = batch['velocity_obs'][:,:self.T_obs]
        nPosition_obs = batch['position_obs'][:,:self.T_obs]
        nAction_obs = batch['action_obs'][:,:self.T_obs]
        B = 1

        # encoder vision features
        image_features = self.nets['vision_encoder'](
            nimage_obs.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage_obs.shape[:2],-1)
        # (B,obs_horizon,D)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nVelocity_obs, nPosition_obs, nAction_obs], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, self.T_pred, self.diffusion_out_dim), device=self.device)
        naction = noisy_action
        self.noise_scheduler.set_timesteps(self.noise_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.nets['noise_predictor'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        nAction_obs = nAction_obs.detach().to('cpu').numpy()
        nPosition_obs = nPosition_obs.detach().to('cpu').numpy()
        nVelocity_obs = nVelocity_obs.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        # (B, pred_horizon, action_dim)
        naction = naction[0]

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1
            },
        }
    
    def load_checkpoint(self, checkpoint_path: str):

        self.model_fromFile = torch.load(checkpoint_path, map_location='cuda')

        ema_net = CreateDenoisingNet(
            self.T_pred,
            self.T_obs,
            self.T_action,
            512,
            7,
            self.global_cond_dim,
            self.diffusion_out_dim,
            self.noise_steps,
            device=self.device
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_steps,
            beta_schedule='squaredcos_cap_v2',    # the choice of beta schedule has a big impact on performance
            clip_sample=True,    # clip output to [-1,1] to improve stability
            prediction_type='epsilon'    # our network predicts noise (instead of denoised action)
        )

        self.ema_net = ema_net.load_state_dict(self.model_fromFile)
        print('Pretrained weights loaded.')

    def plt2tsb(figure, writer, fig_name, niter):
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)

        # Open the image and convert to RGB, then to Tensor
        image = Image.open(buf).convert('RGB')
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

        # Add the image to TensorBoard
        writer.add_image(fig_name, image_tensor, niter)

