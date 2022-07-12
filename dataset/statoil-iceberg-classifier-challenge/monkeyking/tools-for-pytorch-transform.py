#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:19:59 2017

@author: peng
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
import random
import PIL
from PIL import Image, ImageOps
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

'''
data=full_img_val
labels=val['is_iceberg']
angle=angle_val
idx=0
'''
class read_data(Dataset):
    """total dataset."""

    def __init__(self, data, labels, angle,transform=None):

        self.data= data
        self.labels = labels
        self.transform = transform
        self.angle = angle

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx,:,:,:], 'labels': np.asarray([self.labels.values[idx]]), 'angle': np.asarray([self.angle.values[idx]])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, angle = sample['image'], sample['labels'], sample['angle']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = image.astype(float)/255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).long(),
                'angle': torch.from_numpy(angle).float()}

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle = sample['image'], sample['labels'], sample['angle']

        
        if random.random() < 0.5:
            image=np.flip(image,1)
        
        return {'image': image, 'labels': labels,'angle':angle}
    
class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle = sample['image'], sample['labels'], sample['angle']
        
        if random.random() < 0.5:
            image=np.flip(image,0)
        
        return {'image': image, 'labels': labels, 'angle':angle} 

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        img=tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels'], 'angle': tensor['angle']}  