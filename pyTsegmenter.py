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
import time


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
    
    def __init__(self, in_channel=3,output_dim=2):
        super(SegmenterCNN, self).__init__() 
        self.input_dim = in_channel
        self.output_dim = output_dim
        
        self.conv1_1 = torch.nn.Conv2d(in_channel,30,3,1,1)
        self.conv1_2 = torch.nn.Conv2d(30,30,3,1,1)
        
        self.conv2_1 = torch.nn.Conv2d(30,40,3,1,1)
        self.conv2_2 = torch.nn.Conv2d(40,40,3,1,1)
        
        self.conv3_1 = torch.nn.Conv2d(40,40,3,1,1)
        self.conv3_2 = torch.nn.Conv2d(40,40,3,1,1)
        
        self.conv4_1 = torch.nn.Conv2d(40,50,3,1,1)
        self.conv4_2 = torch.nn.Conv2d(50,50,3,1,1)
        
        self.conv5_1 = torch.nn.Conv2d(50,100,3,1,1)
        self.conv5_2 = torch.nn.Conv2d(100,100,3,1,1)
        
        self.final_layer = torch.nn.Conv2d(100,1,3,1,2)
        
    def forward(self,x):
    
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_2(x1))
        x3 = F.relu(self.conv2_1(x2))
        x4 = F.relu(self.conv2_2(x3))
        x5 = F.relu(self.conv3_1(x4))
        x6 = F.relu(self.conv3_2(x5))
        x7 = F.relu(self.conv4_1(x6))
        x8 = F.relu(self.conv4_2(x7))
        x9 = F.relu(self.conv4_1(x8))
        x10 = F.relu(self.conv4_2(x9))
        output_map = F.softmax(self.final_layer(x10))
        
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

class Segmenter(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        
        # load dataset
        # networks init
        self.net = SegmenterCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        def DSC(logits,labels):
            pred_1 = logits.view(2,300,300)[1]
            labels_1 = labels.view(2,300,300)[1]
            
            intersection = torch.sum(pred_1*labels_1)
            union = torch.sum(pred_1) + torch.sum(labels_1)
            
            return (2*intersection)/union
    
        self.loss = DSC()
    
        
    def train(self, batch_size=4, n_epochs=50):
        net = self.net
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate=", self.lr)
        print("=" * 30)
        
        #Get training data
        train_loader = get_train_loader(batch_size)
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
                    
                
    



