'''
SCRIPT TO GENERATE HUBMAP DATASET FROM THE 8 TRAINING IMAGES PROVIDED
THIS SCRIPT GENERATES 5185, 256x256 IMAGES
'''
import collections
import json
import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile as tiff

import albumentations as A

from skimage.measure import label, regionprops
import cv2

TRAIN_PATH = "../input/hubmap-kidney-segmentation/train/"
TEST_PATH = "../input/hubmap-kidney-segmentation/test/"
FINAL_TRAIN_IMAGE = "./final_train/image/" ## FINAL FOLDER WHERE IMAGES WILL BE STORED
FINAL_TRAIN_MASK = "./final_train/mask/" ## FINAL FOLDER WHERE CORRESPONDING MASKS WILL BE STORED
IMG_SIZE = 256 ## ONLY CHANGE THIS VARIABLE IF YOU NEED TO CREATE FINAL IMAGES WITH HIGHER RESOLUTION - CAN BE 512 OR IF YOU ARE FEELING BRAVE 1024
tile_size = 512
os.makedirs(FINAL_TRAIN_IMAGE, exist_ok=True)
os.makedirs(FINAL_TRAIN_MASK, exist_ok=True)

## Extra dataset information about all the training images
dataset_info = pd.read_csv("../input/hubmap-kidney-segmentation/HuBMAP-20-dataset_information.csv")
print(dataset_info.shape)
dataset_info.head(13).T

## Training dataset information
train_df = pd.read_csv("../input/hubmap-kidney-segmentation/train.csv")
print(train_df.shape)
train_df.head().T # Contains the training image id and the RLE

## We need to decode the mask from encoding column of train.csv
## https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

## Tiling the image and mask for easy read and write
## https://www.kaggle.com/marcosnovaes/hubmap-read-data-and-build-tfrecords
def show_tile_and_mask(baseimage, mask, tile_size, tile_col_pos, tile_row_pos):
    start_col = tile_col_pos*tile_size
    end_col = start_col + tile_size
    start_row = tile_row_pos * tile_size
    end_row = start_row + tile_size
    tile_image = baseimage[start_col:end_col, start_row:end_row,:]
    tile_mask = mask[start_col:end_col, start_row:end_row]
    return tile_image, tile_mask
    
def get_tile(baseimage, tile_size, tile_col_pos, tile_row_pos):
    start_col = tile_col_pos*tile_size
    end_col = start_col + tile_size
    start_row = tile_row_pos * tile_size
    end_row = start_row + tile_size
    tile_image = baseimage[start_col:end_col, start_row:end_row,:]
    return tile_image

def get_tile_mask(baseimage, tile_size, tile_col_pos, tile_row_pos):
    start_col = tile_col_pos*tile_size
    end_col = start_col + tile_size
    start_row = tile_row_pos * tile_size
    end_row = start_row + tile_size
    tile_image = baseimage[start_col:end_col, start_row:end_row]
    return tile_image

def show_tile_dist(tile):
    fig, ax = plt.subplots(1,2,figsize=(20,3))
    #ax[0].set_title("Tile ID = {} Xpos = {} Ypos = {}".format(img_mtd['tile_id'], img_mtd['tile_col_pos'],img_mtd['tile_row_pos']))
    ax[0].imshow(tile)
    ax[1].set_title("Pixelarray distribution");
    sns.distplot(tile.flatten(), ax=ax[1]);

augment = A.Compose([
    A.ShiftScaleRotate(p=0.75),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.CLAHE(p=0.75),
    A.RandomRotate90(p=0.5),
    A.InvertImg(p=0.2),
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(75, 100), height=256, width=256, p=0.20),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ], p=1),
    A.ElasticTransform(p=0.5),
    A.ISONoise(p=1),
    A.RandomBrightnessContrast(p=0.5),
], p=1)

def convert_to_images(image, image_data, pos):
    '''
    Function taking the image id, image data and pos in df and converting to smaller images of all glomeruli
    '''
    ## Looking into a single training image
    width = int(dataset_info.loc[dataset_info['image_file'] == image]["width_pixels"])
    height = int(dataset_info.loc[dataset_info['image_file'] == image]["height_pixels"])
    print(image, width, height)
    ## Getting the mask of the image
    mask = rle2mask(train_df.iloc[pos, 1], (width, height)) # Call the RLE2Mask function
    ## Identify all the coordinates of the glomeruli in this image
    lbl = label(mask) 
    props = regionprops(lbl)
    print(len(props), " <-- Number of Glomeruli") ## There are n glomeruli's identified
    bboxes = [] ## Convert all the n items into bounding boxes so we can save these images for training
    for prop in props:
        bboxes.append([prop.bbox[0] - 50, prop.bbox[1] - 50, prop.bbox[2] + 50, prop.bbox[3] + 50]) ## Adding a little bit of extra image run
        
    i = 0
    for bbox in bboxes:
        print(bbox[0], bbox[2], bbox[1], bbox[3])
        augmented = augment(image=image_data[bbox[0]:bbox[2], bbox[1]:bbox[3], :], \
                            mask=mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]) ## Apply first transformation
        augmented2 = augment(image=image_data[bbox[0]:bbox[2], bbox[1]:bbox[3], :], \
                            mask=mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]) ## Apply second transformation
        augmented3 = augment(image=image_data[bbox[0]:bbox[2], bbox[1]:bbox[3], :], \
                            mask=mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]) ## Apply third transformation

        # Write all transformations and it's original maps to folder
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_aug1_' + str(i) + '.png', \
                    cv2.resize(augmented['image'], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_aug2_' + str(i) + '.png', \
                    cv2.resize(augmented2['image'], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_aug3_' + str(i) + '.png', \
                    cv2.resize(augmented3['image'], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_original_' + str(i) + '.png', \
                    cv2.resize(image_data[bbox[0]:bbox[2], bbox[1]:bbox[3], :], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_aug1_' + str(i) + '.png', \
                    cv2.resize(augmented['mask']*255.0, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_aug2_' + str(i) + '.png', \
                    cv2.resize(augmented2['mask']*255.0, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_aug3_' + str(i) + '.png', \
                    cv2.resize(augmented3['mask']*255.0, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_original_' + str(i) + '.png', \
                    cv2.resize(mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]*255.0, (IMG_SIZE, IMG_SIZE))); i+=1
        

def convert_to_images_big(image, image_data, pos):
    '''
    Function taking the image id, image data and pos in df and converting to smaller images of all glomeruli
    '''
    width = int(dataset_info.loc[dataset_info['image_file'] == image]["width_pixels"])
    height = int(dataset_info.loc[dataset_info['image_file'] == image]["height_pixels"])
    print(image, width, height)
    ## Getting the mask of the image
    mask = rle2mask(train_df.iloc[pos, 1], (width, height)) # Call the RLE2Mask function
    with open(TRAIN_PATH + image.split('.')[0] + '.json') as f:
        data = json.load(f)
    print(data[0], len(data))
    
    all_starting_pos = []
    ov_len = len(data)
    for i in range(ov_len):
        all_starting_pos.append(data[i]['geometry']['coordinates'][0][0]) ## We target the coordinates of the json file
    i = 0
    # for i in all_starting_pos:
    for index in all_starting_pos:
        print(index)
        new_img, new_mask = show_tile_and_mask(image_data, mask, tile_size, int(index[1]/tile_size), int(index[0]/tile_size))
        augmented = augment(image=new_img, mask=new_mask) ## Apply first transformation
        augmented2 = augment(image=new_img, mask=new_mask) ## Apply second transformation

        # Write all transformations and it's original maps to folder
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_aug1_' + str(i) + '.png', \
                    cv2.resize(augmented['image'], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_aug2_' + str(i) + '.png', \
                    cv2.resize(augmented2['image'], (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_IMAGE + image.split('.')[0] + '_original_' + str(i) + '.png', \
                    cv2.resize(new_img, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_aug1_' + str(i) + '.png', \
                    cv2.resize(augmented['mask']*255.0, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_aug2_' + str(i) + '.png', \
                    cv2.resize(augmented2['mask']*255.0, (IMG_SIZE, IMG_SIZE)))
        cv2.imwrite(FINAL_TRAIN_MASK + image.split('.')[0] + '_mask_original_' + str(i) + '.png', \
                    cv2.resize(new_mask*255.0, (IMG_SIZE, IMG_SIZE))); i+=1
        
image = train_df.iloc[0, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
convert_to_images(image, image_data, 0)

## Training image number 2
image = train_df.iloc[1, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
convert_to_images(image, image_data, 1)

## Now onwards using tiles to get the masks and tiles as these files are a lot larger than the two previous ones
## Training image number 3
image = train_df.iloc[2, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
convert_to_images_big(image, image_data, 2)

## Onwards!
## Training image number 4
image = train_df.iloc[3, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
convert_to_images_big(image, image_data, 3)

## Training image number 5
image = train_df.iloc[4, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
image_data = image_data[0][0].transpose(1, 2, 0) # Realign the image
convert_to_images(image, image_data, 4)

## Training image number 6
image = train_df.iloc[5, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
image_data = image_data[0][0].transpose(1, 2, 0) # Realign the image
convert_to_images_big(image, image_data, 5)

## Training image number 7
image = train_df.iloc[6, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
image_data = image_data[0][0].transpose(1, 2, 0) # Realign the image
convert_to_images_big(image, image_data, 6)

## Training image number 8! -- Final ONE
image = train_df.iloc[7, 0] + ".tiff"
image_data = tiff.imread(TRAIN_PATH + image)
print(image_data.shape)
image_data = image_data[0][0].transpose(1, 2, 0) # Realign the image
convert_to_images_big(image, image_data, 7)