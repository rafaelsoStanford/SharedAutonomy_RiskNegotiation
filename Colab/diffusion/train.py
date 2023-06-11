import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint

import zarr
import numpy as np

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int, 
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx, 
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class LoadCarRacingData(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 16 , T_obs=4, T_pred=8 , T_act =1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_act = T_act

    def setup(self):
        dataset_root = zarr.open(self.data_dir, "r")

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # (N,3,96,96) Meaning N images, 3 channels, 96x96 pixels

        # (N, D)
        train_data = {
            'position': dataset_root['data']['position'][:], # (T,2)
            # velocity of car
            'velocity': dataset_root['data']['velocity'][:], # (T,2)
            # Action taken by "human" 
            'h_action': dataset_root['data']['h_action'][:], #(T,3)
            # Action taken by control policy ("Autonomous policy")
            'action': dataset_root['data']['action'][:] #(T,3)
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length= self.T_pred,
            pad_before= self.T_obs-1,
            pad_after= self.T_act-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
        
        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.T_pred,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.T_obs,:]
        nsample['velocity'] = nsample['velocity'][:self.T_obs,:]
        nsample['position'] = nsample['position'][:self.T_obs,:]
        nsample['h_action'] = nsample['h_action'][:self.T_obs,:]
        return nsample

    def train_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True, 
            # don't kill worker process afte each epoch
            persistent_workers=True 
        )




def main(n_epochs=100, AMP=True, batch_size=16):
    # ===========data===========
    # Load Dataset using Pytorch Lightning DataModule
    dataset = LoadCarRacingData(data_dir="./data/multipleDrivingBehaviours.zarr", batch_size=batch_size)
    dataset.setup()
    dataloader = dataset.train_dataloader()

    
    # # ===========model===========
    # #model = Diffusion(img_size=img_size)
    
    # # -----PL configs-----
    # tensorboard = pl_loggers.TensorBoardLogger(save_dir="Logs/TrainLogs",name='',flush_secs=1)
    # early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-6, patience=n_epochs)   
    # checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
    #                                       save_top_k=-1,          # Save all checkpoints
    #                                       every_n_epochs=1,               # Save every epoch
    #                                       save_on_train_epoch_end=True,
    #                                       verbose=True)

    # # train model
    # trainer = pl.Trainer(accelerator='gpu', devices=1, precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
    #                      callbacks=[early_stop_callback, checkpoint_callback],
    #                      logger=tensorboard, profiler="simple", val_check_interval=0.25, 
    #                      accumulate_grad_batches=1, gradient_clip_val=0.5)

    
    # trainer.validate(model=model, dataloaders=test_dataloader)
    # trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

if __name__ == "__main__":
    main()