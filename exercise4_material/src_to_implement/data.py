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
        #print(self.data.iloc[0:10])
        pass
    def __len__(self):
        return len(self.data)

    def __getitem__(self , index):
        print(self.data.iloc[index])
        image  =  imread(self.data.iloc[index]['filename'])
        image  = gray2rgb(image)
        #print(image)
        item = (image , self.data.iloc[index]['crack'] , self.data.iloc[index]['inactive'])

        return