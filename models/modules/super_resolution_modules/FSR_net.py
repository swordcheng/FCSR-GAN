import torch
import torch.nn as nn

from models.modules.super_resolution_modules.FSR_modules.coarse_SR_net import CoarseSR
from models.modules.super_resolution_modules.FSR_modules.fine_SR_net import FineSR
from models.modules.super_resolution_modules.FSR_modules.prior_estimation_net import PriorEstimation


class FSRNet(nn.Module):

    def __init__(self):
        
        super(FSRNet, self).__init__()
        self.coarse_SR = CoarseSR()
        self.fine_SR = FineSR()
        self.prior_estimation = PriorEstimation()

    def forward(self, img):

        img_coarse = self.coarse_SR(img)

        landmark, face_parsing = self.prior_estimation(img_coarse)

        img_fine = self.fine_SR(img_coarse, torch.cat((landmark, face_parsing), dim=1))

        return img_coarse, landmark, face_parsing, img_fine
