# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:42:23 2018

@author: hugol
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import cv2
import os
import pandas as pd


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
            label=[1,0]
        else:
            img_name = self.root_dir+'/X_T1/'+self.liste_T[idx]
            label=[0,1]
        image = io.imread(img_name)
        sample_dis = {'image': image,'segment': 0,'source':label}
        
        
        img_name_S = self.root_dir+'/X_S/'+self.liste_S[idx]
        img_name_seg_S = self.root_dir+'/Y_S/'+self.liste_seg_S[idx]
        image_S = io.imread(img_name_S)
        image_seg_S = io.imread(img_name_seg_S)
        sample_S = {'image': image_S, 'segment': image_seg_S}


        if self.transform:
            sample_dis = self.transform(sample_dis)
            sample_S =self.transform(sample_S)

        return sample_dis,sample_S



def resize(sample):
    image = cv2.resize(sample['image'],(150,150)).reshape(3,150,150)/255
    
    sample['image'] = image
    if not isinstance(sample['segment'],int):
        segmentation = cv2.resize(sample['segment'],(132,132))
        segmentation = np.array((255-segmentation,segmentation)).reshape(2,132,132)/255
        sample['segment'] = segmentation

    return sample