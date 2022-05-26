import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size =  batch_size
        self.image_size = image_size
        
        # Augmentation flags
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle


        self.currentEpoch = 0
        self.counter = 0

        with open(self.label_path, 'r') as f:
            self.labels_dict = json.load(f) # Dict map between images and its labels

        self.arrOfpairs = {'images' : [] , 'labels' : []}
        for currfile in os.listdir(self.file_path):
            img = np.resize(np.load(os.path.join(self.file_path, currfile)) , (self.image_size[0],self.image_size[1],self.image_size[2]))
            self.arrOfpairs['images'].append(self.augment(img))
            self.arrOfpairs['labels'].append(self.labels_dict[currfile.split('.')[0]])


        self.pairsDF = pd.DataFrame(data=self.arrOfpairs)

        if self.shuffle:
                self.pairsDF=self.pairsDF.sample(frac=1)


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        next_counter = self.counter+self.batch_size
        self.counter = self.counter%self.pairsDF.shape[0]

        if next_counter > self.pairsDF.shape[0]:
            next_counter = next_counter%self.pairsDF.shape[0]
            pairs1 = self.pairsDF[self.counter:self.pairsDF.shape[0] ]
            pairs2 = self.pairsDF[:(next_counter%self.pairsDF.shape[0]) ]
            pairs = pd.concat([pairs1 , pairs2])
            self.currentEpoch +=1
            

        else:
            pairs = self.pairsDF[self.counter:next_counter]
       

        self.counter = next_counter
        images = np.stack(pairs['images'])
        labels = np.stack(pairs['labels'])

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.rotation :
            img = np.rot90(img)
        
        if self.mirroring :
            img = np.fliplr(img)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.currentEpoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images , labels  =  self.next()
        for i in range(len(images)) :
            image = images[i]
            label = labels[i]
            img = Image.fromarray(image, 'RGB')
            img.show()
            print("Label : ", label)
        return 0

