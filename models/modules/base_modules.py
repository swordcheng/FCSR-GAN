import torch.nn as nn


class ModuleFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(module_name, *args, **kwargs):
        if  module_name == 'pconv':
            from models.modules.inpainting_modules.PConv_net import PConvUNet
            network = PConvUNet(*args, **kwargs)
            print(network)
        elif module_name == 'fsrnet':
            from models.modules.super_resolution_modules.FSR_net import FSRNet
            network = FSRNet()
        elif module_name == 'd':
            from models.modules.discrininator.global_d import Discriminator
            network = Discriminator()
        elif module_name == 'vgg':
            from models.modules.aux_modules.vgg import VGG16FeatureExtractor
            network = VGG16FeatureExtractor()
        elif module_name == 'fp':
            from models.modules.aux_modules.segnet import SegNet
            network = SegNet(num_classes=9)
        else:
            raise ValueError("Module %s not recognized." % module_name)

        print("Module %s was created" % module_name)

        return network


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
        self._name = 'BaseModule'

    @property
    def name(self):
        return self._name




