import torch
from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # self.encoder = resnet18(weights=True)
        # self.encoder = nn.Linear(16, 16)
        # classifier
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Linear(512, 10)

    def forward(self, batch_dict):

        x = batch_dict['data']
        # predict the probability of each class
        x = self.encoder(x)

        result_dict = {
            'pred': x,
        }

        return result_dict

    def print_info(self, logger):

        logger.info('{} info: '.format(self._get_name()))
        n_params = sum([param.nelement() for param in self.parameters()]) * 4 / 1000 / 1000
        logger.info('Total parameters: {:4f}M'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()]) * 4 / 1000 / 1000
        logger.info('encoder parameters: {:4f}M'.format(n_params))