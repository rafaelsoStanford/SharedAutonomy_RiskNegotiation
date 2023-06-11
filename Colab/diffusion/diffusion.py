import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint



class Diffusion(pl.LightningModule):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256,img_channels=1, lr=1e-4):
        super().__init__()

        ### Define parameters
        
        ### Define model which will be a simplifed 1D UNet
   
    
    def training_step(self, batch, batch_idx):
        return loss
    
    def validation_step(self, batch, batch_idx):
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/Val_loss",
                "frequency": 1
            },
        }
    
    
    def on_train_epoch_end(self):
        pass

