import torch.nn as nn

from models.modules.super_resolution_modules.FSR_modules import res_block


class CoarseSR(nn.Module):

    def __init__(self):

        super(CoarseSR, self).__init__()

        self._name = 'Coase_SR'

        self.layers = []

        self.layers.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU(inplace=True))

        for i in range(3):
            self.layers.append(res_block.ResidualBlock(64, 64))

        self.layers.append(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*self.layers)

    def forward(self, img):

        return self.net(img)