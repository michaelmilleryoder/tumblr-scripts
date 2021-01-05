import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        c = copy.deepcopy
        self.after_res = nn.Sequential(
                c(resnet152.fc),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1000, 2)
            )

    def forward(x):
        return self.after_res(x)
