import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PreprocessModel import ResnetPre as res
from torchvision import transforms

from PIL import Image
import os


image_dir = "/usr0/home/yansenwa/courses/11747/web/test/"
store_dir = "/usr0/home/yansenwa/courses/11747/test_res/"
if not os.path.isdir(store_dir):
    os.mkdir(store_dir)

image_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

net = res()
net.to("cuda:3")

dirs = os.listdir(image_dir)
num = 0
with torch.no_grad():
    for iname in dirs:
        ipath = os.path.join(image_dir, iname)
        img = Image.open(ipath, 'r')
        img = img.convert("RGB")
        img_tensor = image_transforms(img)
        feature_before_fc = net(img_tensor.unsqueeze(0).to("cuda:3")).squeeze(0).to("cpu")
        np.save(os.path.join(store_dir, iname.split('.')[0]), feature_before_fc)
        num += 1
        print(num)
print("total image number: %d" % num)
