import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, ypred, ytgt):

        loss = self.loss_fn(ypred, ytgt)

        return loss