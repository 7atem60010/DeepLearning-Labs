import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.cnv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2)
        self.res1 = ResBlock(64, 64, 1)
        self.res2 = ResBlock(64, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.avg = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sig = nn.Sigmoid()

    def forward(self , input ):
        out = self.cnv1(input)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.avg(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.sig(out)

        return out

class ResBlock(nn.Module):
    def __init__(self , in_channels , out_channels , stride ):
        super(ResBlock, self).__init__()
        self.cnv_res_block = nn.Conv2d(in_channels, out_channels, 3, padding=(1,1) ,stride = stride)
        self.bn_res_block = nn.BatchNorm2d(out_channels)
        self.relu_res_block = nn.ReLU()

        self.cnv_res_block2 = nn.Conv2d(out_channels, out_channels, 3 , padding=(1,1))
        self.con1 = nn.Conv2d(in_channels , out_channels, 1 , stride)


    def forward(self , input):

        out = self.cnv_res_block(input)
        out = self.bn_res_block(out)
        out = self.relu_res_block(out)
        out = self.cnv_res_block2(out)
        out = self.bn_res_block(out)

        input = self.con1(input)
        input = self.bn_res_block(input)
        out = out + input

        out = self.relu_res_block(out)

        return out



