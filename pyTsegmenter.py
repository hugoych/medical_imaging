# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:46:14 2018

@author: hugol
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os


class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        seg_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 1])
        image = cv2.imread(img_name+'.jpg',0)
        segmentation = cv2.imread(seg_name + '.jpg',0)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        image = cv2.resize(image,(300,300))
        segmentation = cv2.resize(segmentation,(300,300))
        segmentation = np.array((255-segmentation,segmentation)).reshape(300,300,2)/255
        
        sample = {'image': image, 'segmentation': segmentation}

        if self.transform:
            sample = self.transform(sample)

        return sample




class SegmenterCNN(torch.nn.Module):
    
    def __init__(self, in_channel=3):
        super(SegmenterCNN, self).__init__() 
        self.conv1_1 = torch.nn.Conv2d(in_channel,30,3,1,1)
        self.conv1_2 = torch.nn.Conv2d(30,30,3,1,1)
        
        self.conv2_1 = torch.nn.Conv2d(40,40,3,1,1)
        self.conv2_2 = torch.nn.Conv2d(40,40,3,1,1)
        
        self.conv3_1 = torch.nn.Conv2d(50,50,3,1,1)
        self.conv3_2 = torch.nn.Conv2d(50,50,3,1,1)
        
        self.conv3_1 = torch.nn.Conv2d(100,100,3,1,1)
        self.conv3_2 = torch.nn.Conv2d(100,100,3,1,1)
        
        self.final_layer = torch.nn.Conv2d(100,1,3,1,1)
        
    def forward(self,x):
    
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.softmax(self.final_layer(x))
        
        return np.round(x)
    
    def outputSize(in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)
        
    
    
def createLossAndOptimizer(net, learning_rate=0.001):

#Loss function
    def DSC(logits,labels):
        pass
        
#Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return(loss, optimizer)

import time

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
print("Training finished, took {:.2f}s".format(time.time() - training_start_time))