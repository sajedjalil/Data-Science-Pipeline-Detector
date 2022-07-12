# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:14:02 2016

@author: amovschin
"""


from __future__ import print_function

import os
import numpy as np
import cv2
import random as rd
from subprocess import check_output


def get_nb_images(data_path):
    images = os.listdir(data_path)
    total = int(len(images) / 2)
    return total
    

def get_indices(maxi, ratio):
    nb = int(round(maxi * ratio))
    return rd.sample(list(np.arange(maxi)), nb)

def get_all_names_without_mask(data_path):
    images = os.listdir(data_path)
    im_names = list()
    for image_name in images:
        if 'mask' in image_name:
            continue
        if not ('.tif' in image_name):
            continue
        im_names.append(image_name)
    return im_names

def get_subset_names_wo_mask(data_path, indices):
    all_names = get_all_names_without_mask(data_path)
    subset_names = [all_names[n] for n in indices]
    return subset_names

    
def create_rotated_imgs(data_path, ratio, max_angle):
    # get the names of images to be processed
    nb_im = get_nb_images(data_path)
    ind_rotated_imgs = get_indices(nb_im, ratio)
    rotated_names = get_subset_names_wo_mask(data_path, ind_rotated_imgs)
    
    # process images
    for image_name in rotated_names:
        # get image and mask
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        image_rows, image_cols = img.shape
        
        # random rotation of the image and its mask
        angle = rd.randint(-max_angle, max_angle)
        Rotation_matrix = cv2.getRotationMatrix2D((image_cols/2,image_rows/2),angle,1)
        rotated_im = cv2.warpAffine(img,Rotation_matrix,(image_cols, image_rows))
        rotated_im_mask = cv2.warpAffine(img_mask,Rotation_matrix,(image_cols, image_rows))
        
        # filenames for the rotated images
        rot_name = image_name.split('.')[0] + '_rot.tif'
        rot_mask_name = image_name.split('.')[0] + '_rot_mask.tif'
        
        # saving the rotated image and mask
        cv2.imwrite(os.path.join(data_path, rot_name), rotated_im)
        cv2.imwrite(os.path.join(data_path, rot_mask_name), rotated_im_mask)
        

if __name__ == '__main__':
    rd.seed(1234)
    create_rotated_imgs('../input/train', .01, 180)


