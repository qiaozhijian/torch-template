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

        return [1,2,3]

    def __getitem__(self, idx):

        return idx
