import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, cfg, is_train):
        super(SimpleDataset, self).__init__()

        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_samples(self):

        samples = []
        for i in range(100):
            data = np.random.rand(3, 512, 512)
            # one-hot encoding
            label = np.zeros(10)
            label[i % 10] = 1
            samples.append((data, label))

        return samples

    def __getitem__(self, idx):

        batch_dict = {
            'idx': idx,
            'data': self.samples[idx][0].astype(np.float32),
            'label': self.samples[idx][1].astype(np.float32),
        }

        return batch_dict
