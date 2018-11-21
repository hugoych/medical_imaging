# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:58:25 2018

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
        
        self.conv2_1 = torch.nn.Conv2d(30,40,3,1,1)
        self.conv2_2 = torch.nn.Conv2d(40,40,3,1,1)
        
        self.conv3_1 = torch.nn.Conv2d(40,40,3,1,1)
        self.conv3_2 = torch.nn.Conv2d(40,40,3,1,1)
        
        self.conv4_1 = torch.nn.Conv2d(40,50,3,1,1)
        self.conv4_2 = torch.nn.Conv2d(50,50,3,1,1)
        
        self.conv5_1 = torch.nn.Conv2d(50,100,3,1,1)
        self.conv5_2 = torch.nn.Conv2d(100,100,3,1,1)
        
        self.final_layer = torch.nn.Conv2d(100,1,3,1,1)
        
        self.conv_d_1 = torch.nn.Conv2d(230,100,3,3)
        self.conv_d_2 = torch.nn.Conv2d(100,100,3,3,1)
        self.conv_d_34 = torch.nn.Conv2d(100,100,3,1,1)
        self.fully =torch.nn.Linear(100*34*34,2)
        
        
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
        
        y = torch.cat((x4,x6,x8,x10))
        
        y1 = F.relu(self.conv_d_1(y))
        y2 = F.relu(self.conv_d_2(y1))
        y3 = F.relu(self.conv_d_34(y2))
        y4 = F.relu(self.conv_d_34(y3))
        y4 = y4.view(-1,100*34*34)
        out_discriminator = F.relu(self.fully(y4))
    
        
        return output_map,out_discriminator
    
    def outputSize(in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)