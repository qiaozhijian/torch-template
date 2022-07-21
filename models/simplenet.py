import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.encoder = nn.Linear(16, 16)

    def forward(self, x):

        return x

    def print_info(self, logger):

        logger.info('{} info: '.format(self._get_name()))
        n_params = sum([param.nelement() for param in self.parameters()]) * 4 / 1000 / 1000
        logger.info('Total parameters: {:4f}M'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()]) * 4 / 1000 / 1000
        logger.info('encoder parameters: {:4f}M'.format(n_params))