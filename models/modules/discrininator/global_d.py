import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        self._name = 'global_d'

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        assert x.shape[3] == 128, 'error'
        h = self.main(x)
        out_real = self.conv1(h)

        return out_real.squeeze()
