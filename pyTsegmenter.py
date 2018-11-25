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

    def __init__(self,root_dir='Train/Seg_train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
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

class TargetDataset(Dataset):

    def __init__(self,root_dir='Train/Adv_train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.liste = os.listdir(self.root_dir+'/X_T1')
        self.liste_seg = os.listdir(self.root_dir+'/Y_T')
    def __len__(self):
        #modify with list_dir
        return len(self.liste)

    def __getitem__(self, idx):
        img_name = self.root_dir+'/X_T1/'+self.liste[idx]
        img_name_seg = self.root_dir+'/Y_T/'+self.liste_seg[idx]
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

class AdvDataset(Dataset):

    def __init__(self,root_dir='Train/Adv_train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.liste_S = os.listdir(self.root_dir+'/X_S')
        self.liste_seg_S = os.listdir(self.root_dir+'/Y_S')
        self.liste_T = os.listdir(self.root_dir+'/X_T1')
        self.liste_seg_T = os.listdir(self.root_dir+'/Y_T')
        
    def __len__(self):
        #modify with list_dir
        return len(self.liste_S)

    def __getitem__(self, idx):
        p = np.random.rand(1)
        if p>=0.5:
            img_name = self.root_dir+'/X_S/'+self.liste_S[idx]
            img_name_seg = self.root_dir+'/Y_S/'+self.liste_seg_S[idx]
            label=0
        else:
            img_name = self.root_dir+'/X_T1/'+self.liste_T[idx]
            img_name_seg = self.root_dir+'/Y_T/'+self.liste_seg_T[idx]
            label=1
        image = io.imread(img_name)
        image_seg = io.imread(img_name_seg)
        sample = {'image': image, 'segment': image_seg, 'source':label}

        if self.transform:
            sample = self.transform(sample)

        return sample



def resize(sample):
    image = cv2.resize(sample['image'],(150,150)).reshape(3,150,150)/255
    segmentation = cv2.resize(sample['segment'],(132,132))
    segmentation = np.array((255-segmentation,segmentation)).reshape(2,132,132)/255
    sample['image'] = image
    sample['segment'] = segmentation

    return sample


#%% TESTS
        
from skimage.io import imshow
Seg = SegDataset('Train/Seg_train',transform=resize)

a = Seg.__getitem__(166)
#print(os.listdir('Train/Seg_train/X_S')[1])

#b = imread('Train/Seg_train/X_S/IM_000032.jpg')
imshow(a['segment'][0])
        
#%%



class SegmenterCNN(torch.nn.Module):
    
    def __init__(self, in_channel=3,output_dim=1,adv= False):
        super(SegmenterCNN, self).__init__() 
        self.input_dim = in_channel
        self.output_dim = output_dim
        self.adv = adv
        
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
    
        x1 = F.relu(self.conv1_1(x))
        x2 = F.relu(self.conv1_bn(self.conv1_2(x1)))
        
        x3 = F.relu(self.conv2_1(x2))
        x4 = F.relu(self.conv2_bn(self.conv2_2(x3)))
        
        x5 = F.relu(self.conv3_1(x4))
        x6 = F.relu(self.conv3_bn(self.conv3_2(x5)))
        
        x7 = F.relu(self.conv4_1(x6))
        x8 = F.relu(self.conv4_bn(self.conv4_2(x7)))
        
        x9 = F.relu(self.conv5_1(x8))
        x10 = F.relu(self.conv5_bn(self.conv5_2(x9)))
        output_map = F.softmax(self.final_layer(x10),dim=1)
        in_dis = torch.cat((x4, x6, x8, x10), dim=1)
        
        if self.adv:
            return output_map, in_dis
        else:
            return output_map

class DiscriminatorCNN(torch.nn.Module):
    
    def __init__(self,in_channel=230,output_dim=2):
        super(DiscriminatorCNN, self).__init__()
        self.input_dim = in_channel
        self.output_dim = output_dim
        
        self.conv_d_1 = torch.nn.Conv2d(in_channel,100,3,3)
        self.conv_d_2 = torch.nn.Conv2d(100,100,3,3,5)
        self.conv_d_34 = torch.nn.Conv2d(100,100,3,1,1)
        self.conv_bn = torch.nn.BatchNorm2d(100)

        self.fully =torch.nn.Linear(100*18*18,2)
    
    def forward(self,x):
        
        y1 = F.relu(self.conv_bn(self.conv_d_1(x)))
        
        y2 = F.relu(self.conv_bn(self.conv_d_2(y1)))
        y3 = F.relu(self.conv_bn(self.conv_d_34(y2)))
        y4 = F.relu(self.conv_bn(self.conv_d_34(y3)))
        y4 = y4.view(-1,100*32*32)
        out_discriminator = F.relu(self.fully(y4))
        
        return out_discriminator
    
def get_train_loader(dataset,batch_size):
    train_seg_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers=0)
    return train_seg_loader


class Segmenter(object):
    def __init__(self, epoch=20,lr=1e-5,batch_size = 10, dataset=SegDataset('Train/Seg_train',transform=resize), gpu_mode=True):
        # parameters
        self.epoch = epoch
        self.learning_rate = lr
        self.batch_size = batch_size
        self.dataset = dataset
        self.gpu_mode = gpu_mode
        self.input_dim = 300
        
        # load dataset
        # networks init
        self.net = SegmenterCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        
        def DSC(logits,labels):
            pred = logits.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            labels =labels.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            precision = torch.sum(pred*labels)/torch.sum(pred)
            recall = torch.sum(pred*labels)/torch.sum(labels)

            
            return -2*(precision*recall)/(precision+recall)
    
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        
        
    
        
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
        val_loader = torch.utils.data.DataLoader(SegDataset('Val/Seg_val',transform = resize),
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = 0)
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
                #Wrap them in a Variable object
                if self.gpu_mode:
                    inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                    inputs, labels = Variable(inputs), Variable(labels)
                else:
                    inputs, labels = torch.tensor(data['image'], dtype=torch.float) ,torch.tensor(data['segment'], dtype=torch.float)
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
                    #print('data max', outputs[0,0].max(),outputs[0,1].max())
                    
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()
                
            #At the end of the epoch, do a pass on the validation set
            #test()
            total_val_loss = 0
            
            for data in val_loader:
                
                inputs, labels = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                #Wrap them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)
                
                #Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data[0]
                
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    def test(self):
        pred = self.net.cuda()(torch.tensor(a['image'].reshape(1,3,self.input_dim,self.input_dim),dtype=torch.float).cuda())
        imshow(pred.cpu().detach().numpy().reshape(2,self.input_dim-18,self.input_dim-18)[1])
    
    def save(self,PATH_seg):
        torch.save(self.net.state_dict(), PATH_seg)
        
    def load(self,PATH_seg):
        
        self.net.load_state_dict(torch.load(PATH_seg))
        self.net.eval()




class UDA(object):
    def __init__(self, epoch=20,lr_seg=1e-5,lr_adv=0.001,batch_size = 10, dataset=AdvDataset('Train/Adv_train',transform=resize), gpu_mode=True):
        # parameters
        self.epoch = epoch
        self.lr_seg = lr_seg
        self.lr_adv = lr_adv
        self.batch_size = batch_size
        self.dataset = dataset
        self.gpu_mode = gpu_mode
        self.input_dim = 300
        
        # load dataset
        # networks init
        self.seg = SegmenterCNN()
        self.adv = DiscriminatorCNN()
        self.optimizer_seg = optim.Adam(self.seg.parameters(), lr=self.lr_seg)
        self.optimizer_adv = optim.Adam(self.adv.parameters(), lr=self.lr_adv)
        
        def DSC(logits,labels):
            pred = logits.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            labels =labels.view(2,self.batch_size,self.input_dim-18,self.input_dim-18)
            precision = torch.sum(pred*labels)/torch.sum(pred)
            recall = torch.sum(pred*labels)/torch.sum(labels)

            
            return -2*(precision*recall)/(precision+recall)
    
        self.loss_seg = DSC
        
        
        
        
    
        
    def train(self):
        seg = self.seg.cuda()
        adv = self.adv.cuda()
        batch_size=self.batch_size
        n_epochs=self.epoch
        #Print all of the hyperparameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs=", n_epochs)
        print("learning_rate_seg=", self.lr_seg)
        print("learning_rate_adv=", self.lr_adv)
        print("=" * 30)
        
        "LOADER A MODIFER"     
        val_loader = torch.utils.data.DataLoader(AdvDataset('Val/Adv_val',transform = resize),
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = 0)
        #Get training data
        train_loader = get_train_loader(self.dataset, batch_size)
        
        n_batches = len(train_loader)
        

        
        #Create our loss and optimizer functions
        loss_dis , loss_seg = torch.nn.BCELoss(), self.loss_seg 
        optimizer_seg, optimizer_adv = self.optimizer_seg, self.optimizer_adv
        
        def loss_seg_adv(loss_adv,logits,gt,alpha):
            return loss_seg(logits,gt) -alpha*loss_adv
        
        #Time for printing
        training_start_time = time.time()
        
        #Loop for n_epochs
        alpha = 0
        e1=10
        e2=35
        alpha_max = 0.05
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            
            if epoch >= e1 and epoch <= e2:
                alpha = alpha_max*(epoch-e1)/(e2-1)
            
            for i, data in enumerate(train_loader, 0):
                
                #Get inputs
                #Wrap them in a Variable object
                if self.gpu_mode:
                    inputs, labels_seg = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                    label_adv = torch.tensor(data['source'], dtype=torch.float).cuda()
                    inputs, labels, label_adv = Variable(inputs), Variable(labels_seg), Variable(label_adv)
                else:
                    inputs, labels_seg = torch.tensor(data['image'], dtype=torch.float) ,torch.tensor(data['segment'], dtype=torch.float)
                    label_adv = torch.tensor(data['source'], dtype=torch.float)
                    inputs, labels, label_adv = Variable(inputs), Variable(labels_seg), Variable(label_adv)
                
                #Set the parameter gradients to zero
                optimizer_seg.zero_grad()
                optimizer_adv.zero_grad()
                
                #Forward pass, backward pass, optimize
                outputs_seg = seg(inputs)
                outputs_adv = adv(outputs_seg[1:])
                
                
                loss_discriminator = loss_dis(outputs_adv,label_adv)
                loss_adversarial = loss_seg_adv(loss_discriminator,outputs_seg[0],labels_seg,alpha) 
                #loss_size.backward()
                #optimizer.step()
                
                loss_discriminator.backward()
                optimizer_adv.step()
                
                if label_adv == 0:
                    loss_adversarial.backward()
                    optimizer_seg.step()
                
                #print(list(net.parameters())[0].grad.mean())
                
                #Print statistics
                running_loss += loss_adversarial.item()
                total_train_loss += loss_adversarial.item()
                
                #Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                            epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                    #print('data max', outputs[0,0].max(),outputs[0,1].max())
                    
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()
                
            #At the end of the epoch, do a pass on the validation set
            #test()
            total_val_loss = 0
            
            for data in val_loader:
                
                if self.gpu_mode:
                    inputs, labels_seg = torch.tensor(data['image'], dtype=torch.float).cuda() ,torch.tensor(data['segment'], dtype=torch.float).cuda()
                    label_adv = torch.tensor(data['source'], dtype=torch.float).cuda()
                    inputs, labels, label_adv = Variable(inputs), Variable(labels_seg), Variable(label_adv)
                else:
                    inputs, labels_seg = torch.tensor(data['image'], dtype=torch.float) ,torch.tensor(data['segment'], dtype=torch.float)
                    label_adv = torch.tensor(data['source'], dtype=torch.float)
                    inputs, labels, label_adv = Variable(inputs), Variable(labels_seg), Variable(label_adv)
                    
                #Forward pass
                val_outputs_seg = seg(inputs)
                
                
                
                loss_segmenter = loss_seg(val_outputs_seg[0],labels_seg)

                val_loss_size = loss_segmenter(val_outputs_seg, labels)
                total_val_loss += val_loss_size.data[0]
                
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
                    
                
    def test(self,):
        pred = self.net.cuda()(torch.tensor(a['image'].reshape(1,3,self.input_dim,self.input_dim),dtype=torch.float).cuda())
        imshow(pred.cpu().detach().numpy().reshape(2,self.input_dim-18,self.input_dim-18)[1])
    
    def save(self,PATH_seg,PATH_adv):
        torch.save(self.seg.state_dict(), PATH_seg)
        torch.save(self.adv.state_dict(),PATH_adv)
    
    def load(self,PATH_seg,PATH_adv):
                
        self.seg.load_state_dict(torch.load(PATH_seg))
        self.seg.eval()
        
        self.adv.load_state_dict(torch.load(PATH_adv))
        self.adv.eval()

