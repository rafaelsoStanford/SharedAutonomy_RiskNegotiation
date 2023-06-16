import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Unet import *
from Transformer import *

from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import numpy as np
from PIL import Image
import io


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


class Diffusion_Transformer(pl.LightningModule):
    def __init__(self,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # diffusion
            diffusion_state_action_dim = 7,
            global_cond_dim = 512,

            ):
        super().__init__()

        ### Define parameters
        self.T_obs = n_obs_steps
        self.T_pred = horizon
        self.T_action = n_action_steps

        self.diffusion_state_action_dim = diffusion_state_action_dim
        self.global_cond_dim = global_cond_dim

        # Model parameters
        self.noise_steps = 1000
        self.lr = 1e-4
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

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_steps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample = True,
            prediction_type = 'epsilon')

        # self.register_buffer("beta", self.noise_scheduler.betas)
        # self.register_buffer("alpha", self.noise_scheduler.alphas)
        # self.register_buffer("alpha_hat", self.noise_scheduler.alphas_cumprod)

    # ### Initialize schedule and corresponding variables
    # def cosine_beta_schedule(self, s=0.008):
    #     """Cosine schedule from annotated transformers.
    #     """
    #     steps = self.noise_steps + 1
    #     x = torch.linspace(0, self.noise_steps, steps, device=self.device)
    #     alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    #     return torch.clip(betas, 0.0001, 0.9999)
    
    # def noise_signal(self, x_0, noise ,t):
    #     # Input: x_0 shape = (B, T, Target_dim) ie (B, T_pred, 7)
    #     #z = torch.randn_like(x_0) # Std normal gaussain noise with same shape as x_0
    #     x_t = ( torch.sqrt(self.alpha_hat[t , None, None]) *
    #                     x_0 + torch.sqrt(1.0 - self.alpha_hat[t , None, None]) * noise ) #Noisy image
    #     return x_t
    
    

    def sample(self, cond_features: torch.Tensor, cond: torch.Tensor):
        """Sample from the diffusion model.

        Args:
            cond: The global conditioning tensor.
            timestep: The timestep to sample at.

        Returns:
            A tensor of samples.
        """
        with torch.no_grad():
        # Sample noise - (1, T_pred, diffusion_out_dim)
            noise = torch.randn(
                    (1, self.T_pred, self.diffusion_out_dim), device=self.device) # (1, T_pred, diffusion_out_dim)
            
            nPos_0= cond_features[0, 3:5]
            nVel_0 = cond_features[0, 5:7]

            cond = cond.unsqueeze(0) # (1, global_cond_dim)
            nAction = noise # Initialize action output as noise -- (1, T_pred, diffusion_out_dim)
            

            for k in reversed(range(1, self.noise_steps)):
                # predict noise
                noise_pred = self.noise_predictor(
                    sample=nAction, 
                    timestep=k,
                    global_cond=cond
                )



                # inverse diffusion step (remove noise)
                nAction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=nAction
                ).prev_sample

                nAction[0 ,0 , 3:5] = nPos_0
                nAction[0 ,0 , 5:7] = nVel_0


                #nAction = 1 / torch.sqrt(self.alpha[k]) * (nAction - ( (1 - self.alpha[k]) /torch.sqrt( 1 - self.alpha_hat[k]) ) * noise_pred) + torch.sqrt(self.beta[k]) * noise
                

        return nAction # (1, T_pred, diffusion_out_dim)
        
    
    def onepass(self, batch, batch_idx, mode):
        
        # Prepare data
        nimage = batch['image'][:,:self.T_obs]
        nVelocity_obs = batch['velocity_obs'][:,:self.T_obs] # (B, T_obs, 2)
        nPosition_obs = batch['position_obs'][:,:self.T_obs] # (B, T_obs, 2)
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
        # print(nPosition[0,:,:])

        # Concatenate actions and future positions and velocities
        nTarget = torch.cat([nAction, nPosition, nVelocity], dim=-1) # (B, T_pred, 7)
        # sample noise to add to actions
        noise = torch.randn(nTarget.shape, device=self.device) # Same size as the normalized target
        # sample a diffusion iteration for each data point
        t = torch.randint(low=1, high=self.noise_steps, size=(B,),device=self.device)

        # Get noisy action using scheduler (forward process)
        # noisy_actions = self.noise_signal(
        #     nTarget, noise, t) # (B, T_pred, 7)
        noisy_actions = self.noise_scheduler.add_noise(
                    nTarget, noise, t)
        
        if mode == "Val":
            with torch.no_grad():
                predicted_noise = self.nets['noise_predictor'](noisy_actions, t, obs_cond) # Use the UNET to predict noise: (B, T_pred, 7)
        else:
            predicted_noise = self.nets['noise_predictor'](noisy_actions, t, obs_cond)
        
        loss = self.loss(noise, predicted_noise)

        if mode != "Train" and batch_idx == 0:
            writer = self.logger.experiment
            niter  = self.global_step

            fig = plt.figure()
            fig.clf()
            alphas = np.linspace(1, 0.1, 10)

            position_observed = nPosition_obs.cpu().numpy()
            position_gt = nPosition.detach().cpu().numpy()
            action_state_estimated = self.sample(obs_features[0,:,:], cond = obs_cond[0, : ]).detach().cpu().numpy() # obs_cond was flattened to (B, obs_horizon * obs_dim)

            position_estimated = action_state_estimated[:,:,3:5]



            plt.plot(position_observed[0, :, 0], position_observed[0, :, 1], 'b.')
            plt.plot(position_estimated[0, 0, 0], position_estimated[0, 0, 1], c='yellow')
            plt.scatter(position_estimated[0, 1:, 0], position_estimated[0, 1:, 1], s=10, alpha=alphas, c='r')
            plt.plot(position_gt[0, self.T_obs:, 0], position_gt[0, self.T_obs:, 1], 'g.')
            # plt.plot(position_observed[0,:, 0], position_observed[0,:,1],'b.')
            # plt.scatter(position_estimated[0,:,0],position_estimated[0,:,1],s=10, alpha=alphas, c='r') 
            # plt.plot(position_gt[0, self.T_obs: ,0],position_gt[0, self.T_obs: ,1],'g.')

            plt.grid()
            plt.axis('equal')
            plt2tsb(fig, writer, 'Predicted_path', niter)
            #plt.close(fig)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="Train")
        self.log("train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx, mode="Val")
        self.log("val_loss",loss)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }

    

