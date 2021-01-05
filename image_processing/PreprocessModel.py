import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class ResnetPre(nn.Module):
    def __init__(self):
        super(ResnetPre, self).__init__()
        self.resnet152 = models.resnet152(pretrained=True)
        self.resnet152.fc = IdentityLayer()

    def forward(self, x):
        return self.resnet152(x)
