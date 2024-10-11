import os
import math
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import lightning as L

from datamodules.components.transforms import Normalizer
from datamodules.components.ts_dataset import TSDataset

import warnings
warnings.filterwarnings("ignore")


class TimeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        save_dir: str,
        has_val: bool = True,
        # num_train_epochs: int = 1,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        # self.num_train_epochs = num_train_epochs
        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.split_ratio = [0.7, 0.15, 0.15]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
    
    def prepare_data(self):
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            # load the data
            data = np.load(os.path.join(self.data_dir, "signals.npy"))
            target = np.load(os.path.join(self.data_dir, "targets.npy"))
            if self.split_ratio != None:
                num_train = math.ceil(data.shape[0] * self.split_ratio[0])
                num_val = math.ceil(data.shape[0] * self.split_ratio[1])
                
                train_data, train_target = data[:num_train], target[:num_train]
                val_data, val_target = data[num_train:num_train+num_val], target[num_train:num_train+num_val]
                test_data, test_target = data[num_train+num_val:], target[num_train+num_val:]
            
            normalizer = Normalizer(norm_type='standardization')
            normalizer.fit(train_data)

            # normalize the data and transpose it to [num_samples, features, sequence]
            train_data = normalizer.transform(train_data).transpose(0, 2, 1)
            val_data = normalizer.transform(val_data).transpose(0, 2, 1)
            test_data = normalizer.transform(test_data).transpose(0, 2, 1)
            
            self.train_dataset = TSDataset(train_data, train_target)
            self.val_dataset = TSDataset(val_data, val_target)
            self.test_dataset = TSDataset(test_data, test_target)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        # if not self.has_val:
        #     return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
        )


    @property
    def input_size(self):
        return self.train_dataset.data.numpy().shape[1:]

    @property
    def num_classes(self):
        return len(np.unique(self.train_dataset.target))
