import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from model import ResNet

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data_path  ='/tmp/pycharm_project_476/src_to_implement/data.csv'
data = pd.read_csv(data_path, sep=';')
train , test = train_test_split(data , test_size= 0.2)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dl = t.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=1)
val_test_dl = t.utils.data.DataLoader(ChallengeDataset(test, 'val'), batch_size=1)

# create an instance of our ResNet model
# TODO
model  = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
crit = t.nn.BCELoss()
opt = t.optim.SGD(model.parameters() , lr =0.001 , momentum= 0.9)
trainer = Trainer(model, crit , opt , train_dl , val_test_dl , early_stopping_patience=3)


# go, go, go... call fit on trainer
res = trainer.fit(5)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')