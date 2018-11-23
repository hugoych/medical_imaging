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
from skimage import io
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
import time
import pandas as pd



#%%
class SegDataset(Dataset):

    def __init__(self,root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv('ImageName.csv')
        self.root_dir = 'Train/Seg_train'
        self.transform = transform
        self.liste = os.listdir(self.root_dir+'/X_S')
        self.liste_seg = os.listdir(self.root_dir+'/Y_S')
    def __len__(self):
        #modify with list_dir
        return len(self.liste)

    def __getitem__(self, idx):
        img_name = self.root_dir+'/X_S/'+self.liste[idx]
        img_name_seg = self.root_dir+'/Y_S/'+self.liste_seg[idx]
        image = io.imread(img_name)
        image_seg = io.imread(img_name_seg)
        sample = {'image': image, 'segment': image_seg}

        if self.transform:
            sample = self.transform(sample)

        return sample

train_seg_loader = torch.utils.data.DataLoader(SegDataset('Train/Seg_train'),
                                          batch_size = 10,
                                          shuffle = True,
                                          num_workers=1)


def resize(sample):
    image = cv2.resize(sample['image'],(300,300)).reshape(3,300,300)/255
    segmentation = cv2.resize(sample['segment'],(278,278))
    segmentation = np.array((255-segmentation,segmentation)).reshape(2,278,278)/255
    sample = {'image': image, 'segment': segmentation}
    return sample

val_loader = torch.utils.data.DataLoader(SegDataset('Val/Seg_val',transform = resize),
                                         batch_size = 1,
                                         shuffle = True,
                                         num_workers = 0)
#%% TESTS
        
from skimage.io import imshow
Seg = SegDataset('Train/Seg_train',transform=resize)

a = Seg.__getitem__(166)
#print(os.listdir('Train/Seg_train/X_S')[1])

#b = imread('Train/Seg_train/X_S/IM_000032.jpg')
imshow(a['segment'][0])
        
#%%



class SegmenterCNN(torch.nn.Module):
    
    def __init__(self, in_channel=3,output_dim=1):
        super(SegmenterCNN, self).__init__() 
        self.input_dim = in_channel
        self.output_dim = output_dim
        
        self.conv1_1 = torch.nn.Conv2d(in_channel,30,3,1)
        self.conv1_2 = torch.nn.Conv2d(30,30,3,1)
        
        self.conv2_1 = torch.nn.Conv2d(30,40,3,1)
        self.conv2_2 = torch.nn.Conv2d(40,40,3,1)
        
        self.conv3_1 = torch.nn.Conv2d(40,40,3,1)
        self.conv3_2 = torch.nn.Conv2d(40,40,3,1)
        
        self.conv4_1 = torch.nn.Conv2d(40,50,3,1)
        self.conv4_2 = torch.nn.Conv2d(50,50,3,1)
        
        self.conv5_1 = torch.nn.Conv2d(50,100,3,1)
        self.conv5_2 = torch.nn.Conv2d(100,100,3,1)
        
        self.final_layer = torch.nn.Conv2d(100,2,3,1)
        
    def forward(self,x):
    
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x1))
        x3 = F.relu(self.conv2_1(x2))
        x4 = F.relu(self.conv2_2(x3))
        x5 = F.relu(self.conv3_1(x4))
        x6 = F.relu(self.conv3_2(x5))
        x7 = F.relu(self.conv4_1(x6))
        x8 = F.relu(self.conv4_2(x7))
        x9 = F.relu(self.conv5_1(x8))
        x10 = F.relu(self.conv5_2(x9))
        output_map = F.softmax(self.final_layer(x10),dim=1)
        #print(output_map.view(2,1,300,300))
        return output_map

class DiscriminatorCNN(torch.nn.Module):
    
    def __init__(self,in_channel=230,output_dim=2):
        super(DiscriminatorCNN, self).__init__()
        self.input_dim = in_channel
        self.output_dim = output_dim
        
        self.conv_d_1 = torch.nn.Conv2d(in_channel,100,3,3)
        self.conv_d_2 = torch.nn.Conv2d(100,100,3,3,1)
        self.conv_d_34 = torch.nn.Conv2d(100,100,3,1,1)
        self.fully =torch.nn.Linear(100*34*34,2)
    
    def forward(self,x):
        
        y1 = F.relu(self.conv_d_1(x))
        y2 = F.relu(self.conv_d_2(y1))
        y3 = F.relu(self.conv_d_34(y2))
        y4 = F.relu(self.conv_d_34(y3))
        y4 = y4.view(-1,100*34*34)
        out_discriminator = F.relu(self.fully(y4))
        
        return out_discriminator
    
def get_train_loader(dataset,batch_size):
    train_seg_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers=0)
    return train_seg_loader


class Segmenter(object):
    def __init__(self, epoch=20,lr=1e-4,batch_size = 10, dataset=SegDataset('Train/Seg_train',transform=resize), gpu_mode=True):
        # parameters
        self.epoch = epoch
        self.learning_rate = lr
        self.batch_size = batch_size
        self.dataset = dataset
        self.gpu_mode = gpu_mode
        
        # load dataset
        # networks init
        self.net = SegmenterCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        def DSC(logits,labels):
            pred_1 = logits.view(2,self.batch_size,278,278)
            labels_1 = labels.view(2,self.batch_size,278,278)
            
            intersection_1 = torch.sum(pred_1[1]*labels_1[1])
            intersection_0 = torch.sum(pred_1[0]*labels_1[0])
            union_1 = torch.sum(pred_1[1]) + torch.sum(labels_1[1])
            union_0 = torch.sum(pred_1[0]) + torch.sum(labels_1[0])
            
            return (1-(2*intersection_1)/union_1)
    
        self.loss = DSC

        
    
        
    def train(self):
        net = self.net.cuda()
        batch_size=self.batch_size
        n_epochs=self.epoch
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate=", self.learning_rate)
        print("=" * 30)

        #Get training data
        train_loader = get_train_loader(self.dataset, batch_size)
        n_batches = len(train_loader)
        
        #Create our loss and optimizer functions
        loss, optimizer = self.loss, self.optimizer
        
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
                inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                #Wrap them in a Variable object
                if self.gpu_mode:
                    inputs, labels = Variable(inputs, requires_grad = True), Variable(labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                #Set the parameter gradients to zero
                optimizer.zero_grad()

                
                #Forward pass, backward pass, optimize
                outputs = net(inputs)
                loss_size = loss(outputs, labels)
                #loss_size.backward()
                #optimizer.step()
                loss_size.backward()
                
                #print(list(net.parameters())[0].grad.mean())
                optimizer.step()
                #Print statistics
                running_loss += loss_size.item()
                total_train_loss += loss_size.item()
                
                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    print('data max', outputs[0,0].max(),outputs[0,1].max())
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()
                
            #At the end of the epoch, do a pass on the validation set
            total_val_loss = 0
            
            for data in val_loader:
                
                inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                #Wrap them in a Variable object
                
                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data[0]
                
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
                    
                
    



