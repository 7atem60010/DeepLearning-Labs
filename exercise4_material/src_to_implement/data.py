from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self , data , mode):
        self.data  = data
        self.mode  =  mode
        self._transform = tv.transforms.Compose([ tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean,train_std)])


    def __len__(self):
        return len(self.data)

    def __getitem__(self , index):
        image  =  imread(self.data.iloc[index]['filename'])
        image  = gray2rgb(image)
        image  = self._transform(image)
        image = torch.tensor(image)
        label  = torch.tensor((self.data.iloc[index]['crack'] , self.data.iloc[index]['inactive']))
        return (image , label)