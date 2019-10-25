import torch.nn as nn
import torch.nn.functional as F

from models.modules.super_resolution_modules.FSR_modules import res_block


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):  # (_, 4, 128, 4)
        super(Hourglass, self).__init__()
        self.depth = depth      
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)
        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)

        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2

        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class PriorEstimation(nn.Module):

    def __init__(self, block=Bottleneck, num_stacks=2, num_blocks=4):

        super(PriorEstimation, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks

        self.layers = []

        self.layers.append(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True))
        self.layers.append(nn.BatchNorm2d(self.inplanes))
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(self.inplanes, self.num_feats * 2, kernel_size=3, stride=1, padding=1, bias=True))
        self.layers.append(nn.BatchNorm2d(self.num_feats * 2))
        self.layers.append(nn.ReLU(inplace=True))

        for i in range(3):
            self.layers.append(res_block.ResidualBlock(self.num_feats * 2, self.num_feats * 2))

        self.net1 = nn.Sequential(*self.layers)

        self.layers = []

        self.layers.append(Hourglass(block, num_blocks, self.num_feats, 4))
        self.layers.append(Hourglass(block, num_blocks, self.num_feats, 4))
        
        self.layers.append(nn.Conv2d(self.num_feats * 2, self.num_feats, kernel_size=1, stride=1))

        self.net2 = nn.Sequential(*self.layers)

        self.con_landmark = nn.Conv2d(self.num_feats, 81, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()

        self.con_face_parsing = nn.Conv2d(self.num_feats, 11, kernel_size=1, stride=1)

    def forward(self, img):

        feature1 = self.net1(img)
        feature2 = self.net2(feature1)

        landmark = self.sig(self.con_landmark(feature2))
        face_parsing = self.con_face_parsing(feature2)

        return landmark, face_parsing
