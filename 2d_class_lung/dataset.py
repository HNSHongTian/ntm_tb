import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
from torchvision.transforms import transforms
import resnet
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import ImageFile
device = torch.device("cpu")
transforms = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(51, 55)
])


def getImgAndLabel(path1, path2):
    # ntm
    img = []
    for files in os.listdir(path1):
        # print(files)
        # print(0)
        img_x = Image.open(os.path.join(path1,files))
        img.append((img_x, 0))

#tb
    for files in os.listdir(path2):
        # print(files)
        # print(1)
        img_x = Image.open(os.path.join(path2,files))
        img.append((img_x, 1))


    return img


class LiverDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        imgs = getImgAndLabel(path1, path2)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x,label = self.imgs[index]
        if self.transform is not None:
            img_y = self.transform(x)

        return img_y, label

    def __len__(self):
        return len(self.imgs)

