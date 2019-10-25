import torch
import torch.nn as nn

from models.modules.super_resolution_modules.FSR_modules import res_block


class FineSR(nn.Module):

    def __init__(self):
        super(FineSR, self).__init__()

        self._name = 'Fine_SR'

        self.encode_layers = []

        self.encode_layers.append(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1))
        self.encode_layers.append(nn.BatchNorm2d(64))
        self.encode_layers.append(nn.ReLU(inplace=True))

        for i in range(12):
            self.encode_layers.append(res_block.ResidualBlock(64, 64))

        self.encode_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.encode_net = nn.Sequential(*self.encode_layers)
        
        self.decode_layers = []
        self.decode_layers.append(nn.Conv2d(64 + 81 + 11, 64, kernel_size=3, stride=1, padding=1))
        self.decode_layers.append(nn.BatchNorm2d(64))
        self.decode_layers.append(nn.ReLU(inplace=True))

        self.decode_layers.append(nn.Upsample(scale_factor=2))
        self.decode_layers.append(nn.BatchNorm2d(64))
        self.decode_layers.append(nn.ReLU(inplace=True))

        for i in range(3):
            self.decode_layers.append(res_block.ResidualBlock(64, 64))

        self.decode_layers.append(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        self.decode_net = nn.Sequential(*self.decode_layers)

    def forward(self, img, face_prior):

        enconde_feature = self.encode_net(img)
        SR_img = self.decode_net(torch.cat((enconde_feature, face_prior), dim=1))

        return SR_img