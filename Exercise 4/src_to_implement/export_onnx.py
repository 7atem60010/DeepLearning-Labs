import torch as t
from trainer import Trainer
import sys
import torchvision as tv
# xiwjub-xuHty5-rajpot
epoch = int(sys.argv[1])
#TODO: Enter your model here

crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
