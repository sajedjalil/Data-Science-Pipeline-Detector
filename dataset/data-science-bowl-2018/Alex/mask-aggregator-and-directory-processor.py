'''
Created on Jan 16, 2018

Finding the dataset directory layout clunky?
Dislike that the masks are spread across multiple files?
This script fixes both of those problems.
Takes ~20-30 seconds to run (your mileage may vary)

Three steps to run:
1) Download
2) Change ROOT_DIR to the location of your train/test folders
3) Uncomment the call to the main() function

From the original train/test folders, with a format of:

input_folder_name
    id1
        img_folder
            img1
        mask_folder (not present for test dataset)
            mask1_1
            mask1_2
            mask1_3
            ...
    id2
        img_folder
            img2
        mask_folder
            mask2_1
            mask2_2
            mask2_3
            ...

this script produces an output of: 

output_folder_name
    images
        img1
        img2
        ...
    masks (empty for test dataset)
        img1_mask
        img2_mask
        ...
        
The individual masks are aggregated into a grayscale image with a unique label for each mask
E.g. For an img1 with 107 nuclei, img1_mask will have values from 0 (background) to 107

For the test dataset, a masks folder is created, but obviously it remains empty

To use this script with custom/edited masks, the new masks must be in the masks directory,
and they must also match the filename pattern of the other masks.  Otherwise, they will be ignored
when constructing the aggregate mask.

@author: Alex
'''

import numpy as np
from skimage import io
import os
from re import match, sub
from multiprocessing.pool import ThreadPool
from functools import partial
import warnings

#Change this to your local directory where the train/test folders are
ROOT_DIR='../input'

TRAIN_DIR=os.path.join(ROOT_DIR,'stage1_train')
TEST_DIR=os.path.join(ROOT_DIR,'stage1_test')

#Adjust as needed
PREPROCESSING_THREAD_COUNT=16

def preprocess_individual(folder_path,output_folder):
    '''
    Takes an individual image/mask folder and saves image, combined mask to the output directory after processing the masks (if present)
    '''
    img_path=folder_path+'/images'
    if(len(os.listdir(img_path))>1):
        raise Exception('{} has multiple images in the image folder!'.format(folder_path))
    image_name=os.listdir(img_path)[0]
    #Read image for size, also if save is necessary new folder
    image_file=os.path.join(img_path,image_name)
    img=io.imread(image_file)
    #Saves all images in one 'images' folder
    new_image_location=os.path.join(output_folder+'/images',image_name)
    io.imsave(new_image_location,img)
    
    #Combines all masks into a single 16-bit grayscale image with a unique value for each mask
    all_masks=np.zeros((img.shape[:2]),np.int32)
    mask_path=folder_path+'/masks'
    #Masks will not exist for test data
    if(os.path.exists(mask_path)):
        new_mask_location=os.path.join(output_folder,'masks')
        #Start at 1 because 0 is background
        i=1
        for v in os.listdir(mask_path):
            #If you have added custom things to the mask folder, this prevents their inclusion in the combined mask provided the filename does not have the same pattern as the default masks
            if(match('.{64}\.png',v) is not None):
                a=io.imread(os.path.join(mask_path,v))
                all_masks+=np.where(a,i,0)
                i+=1
        mask_name=sub('\.png','_mask.png',image_name)
        #Convert to 16-bit and save
        io.imsave(os.path.join(new_mask_location,mask_name),all_masks.astype(np.uint16))
        assert(np.max(all_masks)==i-1)
    return

def preprocess_data(input_folder,output_folder=None):
    '''
    Takes a dataset folder and processes the folders in parallel, writing to the output folder
    '''
    if(output_folder is None):
        output_folder=input_folder+'_processed'
    if(not os.path.exists(output_folder)):
        os.mkdir(output_folder)
    if(not os.path.exists(output_folder+'/images')):
        os.mkdir(output_folder+'/images')
    #A masks folder will be created for test data inputs, but it will be empty (because test data has no masks)
    if(not os.path.exists(output_folder+'/masks')):
        os.mkdir(output_folder+'/masks')
    files=[os.path.join(input_folder,f) for f in os.listdir(input_folder)]
    #skimage complains about the combined masks having low contrast, this squelches the many warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool=ThreadPool(PREPROCESSING_THREAD_COUNT)
        pool.map(partial(preprocess_individual,output_folder=output_folder), files, chunksize=1)
    return

def main():
    #Remember to change the ROOT_DIR to the location of the training/test folders!
    print('Processing training files')
    preprocess_data(TRAIN_DIR)
    print('Processing testing files')
    preprocess_data(TEST_DIR)
    print('Done!')
    return

#main()