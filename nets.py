# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:45:37 2018

@author: hugol
"""


import torch
import torch.nn.functional as F

class SegmenterCNN(torch.nn.Module):
    
    def __init__(self, in_channel=3,output_dim=132,adv= False):
        super(SegmenterCNN, self).__init__() 
        self.input_dim = in_channel
        self.output_dim = output_dim
        self.adv = adv
        
        #Upsample is confusing but it's downsampling
        self.downsample=torch.nn.Upsample(size=(132, 132), mode='bilinear')
        
        self.conv1_1 = torch.nn.Conv2d(in_channel,30,3,1)
        self.conv1_2 = torch.nn.Conv2d(30,30,3,1)
        self.conv1_bn = torch.nn.BatchNorm2d(30)

        
        self.conv2_1 = torch.nn.Conv2d(30,40,3,1)
        self.conv2_2 = torch.nn.Conv2d(40,40,3,1)
        self.conv2_bn = torch.nn.BatchNorm2d(40)

        
        self.conv3_1 = torch.nn.Conv2d(40,40,3,1)
        self.conv3_2 = torch.nn.Conv2d(40,40,3,1)
        self.conv3_bn = torch.nn.BatchNorm2d(40)
        
        
        self.conv4_1 = torch.nn.Conv2d(40,50,3,1)
        self.conv4_2 = torch.nn.Conv2d(50,50,3,1)
        self.conv4_bn = torch.nn.BatchNorm2d(50)

        
        self.conv5_1 = torch.nn.Conv2d(50,100,3,1)
        self.conv5_2 = torch.nn.Conv2d(100,100,1,1)
        self.conv5_bn = torch.nn.BatchNorm2d(100)

        
        self.final_layer = torch.nn.Conv2d(100,2,1,1)
        
    def forward(self,x):
    
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_bn(self.conv1_2(x)))
        
        x = F.relu(self.conv2_1(x))
        x4 = F.relu(self.conv2_bn(self.conv2_2(x)))
        
        x = F.relu(self.conv3_1(x4))
        x6 = F.relu(self.conv3_bn(self.conv3_2(x)))
        
        x = F.relu(self.conv4_1(x6))
        x8 = F.relu(self.conv4_bn(self.conv4_2(x)))
        
        x = F.relu(self.conv5_1(x8))
        x10 = F.relu(self.conv5_bn(self.conv5_2(x)))
        output_map = F.softmax(self.final_layer(x10),dim=1)
        in_dis = torch.cat((self.downsample(x4), self.downsample(x6), self.downsample(x8), x10), dim=1)
        if self.adv:
            return output_map, in_dis
        else:
            return output_map

class DiscriminatorCNN(torch.nn.Module):
    
    def __init__(self,in_channel=230,output_dim=2):
        super(DiscriminatorCNN, self).__init__()
        self.input_dim = in_channel
        self.output_dim = output_dim
        
        self.conv_d_1 = torch.nn.Conv2d(in_channel,100,3,1,1)
        self.conv_d_2 = torch.nn.Conv2d(100,100,3,1,)
        self.conv_d_34 = torch.nn.Conv2d(100,100,3,1,1)
        
        self.conv_bn = torch.nn.BatchNorm2d(100)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3,stride=3)


        self.fully =torch.nn.Linear(100*14*14,2)
    
    def forward(self,x):
        
        y = F.relu(self.maxpool1(self.conv_bn(self.conv_d_1(x))))
        y = F.relu(self.maxpool1(self.conv_bn(self.conv_d_2(y))))
        y = F.relu(self.conv_bn(self.conv_d_34(y)))
        y = F.relu(self.conv_bn(self.conv_d_34(y)))
        y = y.view(-1,100*14*14)
        out_discriminator = F.softmax(self.fully(y))
        
        return out_discriminator