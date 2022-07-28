import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class ResNet(nn.Module):
    def __init__(self, ):
        super(ResNet, self).__init__()
        self.cnv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn = nn.BatchNorm2d()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool(3, 2)
        self.res1 = nn.ResBlock(64, 64, 1)
        self.res2 = nn.ResBlock(64, 128, 2)
        self.res3 = nn.ResBlock(128, 256, 2)
        self.res4 = nn.ResBlock(256, 512, 2)
        self.avg = nn.GlobalAvgPool()
        self.flatten = nn.Flatten()
        self.fc = nn.FC(512, 2)
        self.sig = nn.Sigmoid()

    def ResNet(self, in, out, stride):


