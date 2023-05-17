from torch.utils.data import DataLoader, Subset
from baseline.DeepSAD.src.base.base_dataset import BaseADDataset
from baseline.DeepSAD.src.base.odds_dataset import ODDSDataset
from .preprocessing import create_semisupervised_setting

import torch
from torch.utils.data import Sampler
import numpy as np


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 dataset: ODDSDataset,
                 batch_size=256, 
                 ano_ratio=0.5, 
                 steps_per_epoch=20, 
                 ):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.steps_per_epoch = steps_per_epoch

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)

        if ano_ratio == 0:
            self.n_outlier = 0
        elif ano_ratio == -1:
            self.n_outlier = len(self.dataset.outlier_idx)
        else:
            self.n_outlier = int(batch_size * ano_ratio)
        self.n_normal = batch_size - self.n_outlier
            

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield batch



class ODDSADDataset(BaseADDataset):

    def __init__(self, data, train):
        super().__init__(self)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # training or testing dataset
        self.train = train

        if self.train:
            # Get training set
            self.train_set = ODDSDataset(data=data, train=True)
        else:
            # Get testing set
            self.test_set = ODDSDataset(data=data, train=False)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, use_sampler=False, sample_ano_ratio=0.5, steps_per_epoch=20):

        if self.train:
            if use_sampler:
                sampler = BalancedBatchSampler(self.train_set, batch_size=batch_size, ano_ratio=sample_ano_ratio, steps_per_epoch=steps_per_epoch)
                batch_size = 1
                shuffle_train = False
            else:
                sampler=None
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, 
                                      batch_sampler=sampler, num_workers=num_workers, drop_last=False if sampler else True)
            return train_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, drop_last=False)
            return test_loader