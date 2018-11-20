#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:16:39 2018

@author: raphael
"""

#medical imaging file to preprocess the data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow
from skimage import util
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import AxesGrid
from time import time
#import cv2
from PIL import ImageFilter

#%%
#import the files of names
names_df = pd.read_csv("ImageName.csv")
names = names_df['ImageId'].values
names_seg = names_df['ImageSegId'].values

n = names.size

for i  in range(n):
    if i<450:
        im = imread('im_resized/'+names[i]+'_resized.jpg')
        im_seg = imread('im_resized/'+names_seg[i]+'_resized.jpg')
        filename = 'Source/X_S/'+names[i]+'.jpg'
        filename_seg = 'Source/Y_S/'+names_seg[i]+'.jpg'
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        imsave(filename,im)
        if not os.path.exists(os.path.dirname(filename_seg)):
            try:
                os.makedirs(os.path.dirname(filename_seg))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        imsave(filename_seg,im_seg)
        
    if i>=450:
        im = imread('im_resized/'+names[i]+'_resized.jpg')
        filename = 'Target/X_T1/'+names[i]+'.jpg'
        filename2 = 'Target/X_T2/'+names[i]+'.jpg'
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        im_noised_r = util.random_noise(im[:,:,1], var = 0.01)
        im_noised_r = np.array(255*im_noised_r,dtype = np.uint8)
        im_final = np.zeros(im.shape,dtype = np.uint8)
        im_final[:,:,1] = im_noised_r.copy()
        im_final[:,:,0] = im[:,:,0].copy()
        im_final[:,:,2] = im[:,:,2].copy()
        imsave(filename,im_final)
        if not os.path.exists(os.path.dirname(filename2)):
            try:
                os.makedirs(os.path.dirname(filename2))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        im_noised_r = util.random_noise(im[:,:,1], var = 0.006)
        im_noised_r = np.array(255*im_noised_r,dtype = np.uint8)

        im_noised_b = util.random_noise(im[:,:,0], var = 0.001)
        im_noised_b = np.array(255*im_noised_b,dtype = np.uint8)

        im_noised_g = util.random_noise(im[:,:,2], var = 0.005)
        im_noised_g = np.array(255*im_noised_g,dtype = np.uint8)
        
        im_final = np.zeros(im.shape,dtype = np.uint8)
        im_final[:,:,0] = im_noised_b.copy()
        im_final[:,:,1] = im_noised_r.copy()
        im_final[:,:,2] = im_noised_g.copy()
        imsave(filename2,im_final)
        #imsave(filename2,im)
#%%


#%%
       
        
#print(k)
#data_separation
## create an image transformation to put 60 percent in S and 40 percent in T
## S is a folder with both segmentation and image
## T only image
## image transformation could ve : transfor to Y,u,v space apply a linear transform on u or v? Analog with GE, SWI
## create B_adv, B_seg for the training 
## create a validation set

#for im_name in names_df['ImageId'].values:
    


