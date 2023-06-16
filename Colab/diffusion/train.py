import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint


from Diffusion import *
from LoadCarRacingData import * 

def main(n_epochs=100, AMP=True, batch_size=32):

    # Parameters:
    T_obs= 16
    T_pred= 32
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

    # ===========data===========
    # Load Dataset using Pytorch Lightning DataModule
    dataset = CarRacingDataModule(data_dir="./data" , batch_size=batch_size,
                                T_obs=T_obs, T_pred=T_pred , T_act =T_act)
    dataset.setup(name='multipleDrivingBehaviours_5Eps_parallel.zarr.zip')
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.val_dataloader()
    
    # # ===========model===========
    diffusion = Diffusion(T_obs=T_obs, T_pred=T_pred , T_action =T_act, global_cond_dim=obs_dim ,diffusion_out_dim= action_dim)
    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="tb_logs/",name='',flush_secs=1)

    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=1e-7, patience=n_epochs)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=5,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)
    # train model
    trainer = pl.Trainer(accelerator='gpu', devices=[0,1], precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=0.5)
    
    trainer.validate(model= diffusion, dataloaders=valid_dataloader)
    trainer.fit(model=diffusion, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    main()