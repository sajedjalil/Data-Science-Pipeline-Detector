# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sys
import dippykit as dip
import numpy    as np
from scipy.stats import rankdata
from skimage.feature import greycomatrix, greycoprops

import cv2
import pysal as ps
import math
import time
from skimage.color import rgb2grey
from scipy.ndimage import zoom
from skimage.morphology import binary_opening, disk, label

import skimage as ski #Library with various image processing algorithms (i.e. segmentation)
import os
import random

#####################
# NATHAN'S FUNCTIONS
#####################

def remove_clouds(input_img, k, D0, grayscale=False, cloud_mask_size=None):
    '''
    USAGE:
        returns RBG image with clouds suppressed
        
        input_img: RBG image
        k: constant bias in Butterworth HPF
        D0: constant to compute the angular frequency in Butterworth HPF
        grayscale: boolean to control if the cloud mask should be computed as the intersect of
                   RBG masks or just the mask of the greyscale of the image
        cloud_mask_size: int to compute the cloud_mask_size x cloud_mask_size downsample of the 
                         image prior to computing cloud mask
    WARNING:
        Using grayscale=False and cloud_mask_size>=256 takes a significant amount of time                 
    '''
    
    # Compute the 2D cloud mask using the 'find_clouds' function    
    clouds = find_clouds(input_img, grayscale, cloud_mask_size)
    
    #Construct the Butterworth HPF
    H = butterworth_HPF(k,D0,input_img.shape)
    
    # Normalize the image to range from 0.0 to 1.0
    input_img = input_img.astype(float)/255.0

    # Construct arrays to put our results into
    new_clouds = np.zeros_like(input_img)
    no_clouds = np.zeros_like(input_img)

    # Itterate over all channels
    for color in range(3):
        # Compute the output in frequency domain
        # The log of the  input image is computed for homomorphic filtering
        # Note: log1p was used to avoid division by 0
        # The FFT of the image in the log domain is compute
        # The image is shifted in frequency so the center is at [0,0]
        # Multiply by the Butterworth HPF
        # Shift the frequency represenation back
        G = np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(np.log1p(input_img[:, :, color])))*H)

        # Our new image is now transformed back to the spatial domain from the frequency/log domain
        # Note: because log1p was used originally, 1 must be subtracted from the result
        new_clouds[:, :, color] = (np.exp(np.fft.ifft2(G))-1).real * clouds
        
        # Plot each channel
        plot = False
        if plot:
            dip.imshow(new_clouds[:, :, color])
            dip.show()
            
        # Ensure the resulting image is between 0.0 and 1.0
        new_clouds[:,:,color] = new_clouds[:,:,color] - np.min(new_clouds[:,:,color])
        new_clouds[:, :, color] = new_clouds[:,:,color] * (np.max(input_img[:,:,color]) /  np.max(new_clouds[:,:,color])) * clouds
        
        # Compute an image that has all the areas except for the cloud covered regions
        no_clouds[:, :, color] = (1 - clouds) * input_img[:, :, color]
        
    # Add the filtered cloud covered regions with the non-cloud covered regions
    output = new_clouds + no_clouds
    return np.abs(output*255).astype(int), clouds


def butterworth_HPF(k, D0, size):
    '''
    USAGE:
        returns a 2D Butterworth HPF in the frequency domain
        the [0,0] point is at the center of the 2D array
        
        k: constant bias in Butterworth equation
        D0: constant to compute the angular frequency
        size: size of the filter to construct, note: only the first two components are used
    '''
    order = 1.
    P = int(size[0]/2)
    Q = int(size[1]/2)
    H = np.zeros((size[0], size[1]),dtype='f')
    
    for u in range(size[0]):
        for v in range(size[1]):
            Duv = math.sqrt((u - P)**2 + (v - Q)**(2*order))
            if Duv !=0:
                H[u,v] = 1 / (1 + k * (D0 / Duv)**(2*order))
    return H 


def find_clouds(input_img, grayscale, reshape):
    '''
    USAGE:
        returns 2D cloud mask in the same x-y shape as the input image
        
        input_img: RBG image to search for clouds
        grayscale: boolean to control if the cloud mask should be computed as the intersect of
                   RBG masks or just the mask of the greyscale of the image
        reshape: int to compute the reshape x reshape downsample of the image prior to computing
                 the moran's I statistic
                 
    WARNING:
        Using grayscale=False and reshape>=256 takes a significant amount of time
    '''
    
    
    X_PIXELS = input_img.shape[0]
    Y_PIXELS = input_img.shape[1]
    
    # Option to compute the cloud mask for the greyscale of the image due to computational cost of rbg
    if grayscale:
        # convert the image to greyscale
        input_img = rgb2grey(input_img)
        
        # downsample due to computational cost of large image
        # WARNING: without reshaping, this will take a long time...
        if reshape is not None:
            img = cv2.resize(input_img, dsize=(reshape, reshape), interpolation = cv2.INTER_CUBIC)
        else:
            img = input_img
            
        # Establish spatial weights to compute Moran's I
        # Compute the Moran's I statistic using the image and established weights
        w = ps.lat2W(img.shape[0], img.shape[1])
        lm = ps.Moran_Local(img, w, transformation='r', permutations=99)
        
        # The .q value determine what quadrant 'e.g. high-high, low-high, etc'
        # Reshape the array to the image shape
        # Keep only the 'high-high' values, i.e. where lm=1
        # Mark everywhere else as 0
        ms = np.reshape(lm.q, img.shape)
        ms = np.ma.masked_where(ms != 1, ms)
        intersect = ms.astype(int)
        intersect[ms!=1] = 0
        
        # Upsample back to the original image size using bi-cubic interpolation
        if reshape is not None:
            intersect = zoom(intersect, (X_PIXELS/reshape, Y_PIXELS/reshape), order=3)
        
        # intersect is the cloud mask
        return intersect
    
    # Otherwise compute the cloud region for each rbg channel
    else:
        if reshape is not None:
            moran_significanceRBG = np.zeros((reshape,reshape,3))
        else:
            moran_significanceRBG = np.zeros(input_img.shape)
            
        for color in range (3):
            # downsample due to computational cost of large image
            # WARNING: without reshaping, this will take a long time...
            if reshape is not None:
                img = cv2.resize(input_img[:,:,color], dsize=(reshape, reshape), interpolation = cv2.INTER_CUBIC)
            else:
                img = input_img[:,:,color]
                
            # Establish spatial weights to compute Moran's I
            # Compute the Moran's I statistic using the image and established weights
            w = ps.lat2W(img.shape[0], img.shape[1])
            lm = ps.Moran_Local(img, w, transformation='r', permutations=99)
        
            # The .q value determine what quadrant 'e.g. high-high, low-high, etc'
            # Reshape the array to the image shape
            moran_significance = np.reshape(lm.q, img.shape)
            
            # Keep only the 'high-high' values, i.e. where lm=1
            # Mark everywhere else as 0
            # hh = 1, ll = 3, lh = 2, hl = 4
            moran_significance = np.ma.masked_where(moran_significance != 1, moran_significance)
            moran_significance = moran_significance.astype(int)
            moran_significance[moran_significance!=1] = 0
            moran_significanceRBG[:,:,color] = moran_significance
        
        # Compute the intersect where all channels detect a cloud
        intersect =np.logical_and(np.logical_and(moran_significanceRBG[:,:,0], moran_significanceRBG[:,:,1]), moran_significanceRBG[:,:,2]).astype(int)
        intersect[intersect != 1] = 0     
            
        # Show the mask for each channel as well as the intersection of all three
        plot = True
        if plot:        
            fig, axarr = dip.subplots(1, 4, figsize=(15, 40))
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[2].axis('off')
            axarr[3].axis('off')
            axarr[0].imshow(moran_significanceRBG[:,:,0])
            axarr[1].imshow(moran_significanceRBG[:,:,1])
            axarr[2].imshow(moran_significanceRBG[:,:,2])
            axarr[3].imshow(intersect)
            axarr[0].set_title("R-component")
            axarr[1].set_title("B-component")
            axarr[2].set_title("G-component")
            axarr[3].set_title("Intersection")
            dip.tight_layout(h_pad=0.1, w_pad=0.1)
            dip.show()
        
        # Upsample back to the original image size using bi-cubic interpolation
        if reshape is not None:
            intersect = zoom(intersect, (X_PIXELS/reshape, Y_PIXELS/reshape), order=3)
            
        # intersect is the cloud mask
        return intersect

###################
# GREG'S FUNCTIONS
###################

def get_texture_properties(im, blk_sz, distances, angles, levels, symmetric, normed):
   # Function Description:
   #
   #   calculates texture properties of an image using glcm;
   #   calculations performed on a block-by-block basis
   #
   # Args:
   #
   #   im        : Description - grayscale image
   #               Type        - 2D ndarray
   #
   #   blk_sz    : Description - block size for glcm calculations
   #               Type        - integer
   #
   #   distances : Description - greycomatrix arg; pixel pair distance offsets
   #               Type        - list
   #
   #   angles    : Description - greycomatrix arg; pixel pair angles (radians)
   #               Type        - list
   #
   #   levels    : Description - greycomatrix arg; number of gray-levels
   #               Type        - integer
   #
   #   symmetric : Description - greycomatrix arg; controls whether output matrix is symmetric
   #               Type        - boolean
   #
   #   normed    : Description - greycomatrix arg; controls whether output matrix is normalized
   #               Type        - boolean
   #
   # Returns:
   #
   #   one 2D ndarray for each texture property;
   #   size of each ndarray depends on:
   #      size of image
   #      size of blocks
   #      size of distances/angles input args

   # shape of glcm blocks
   blk_shape = (blk_sz,blk_sz)

   # functions for calculating texture properties from glcm's
   calc_glcm = lambda blk : greycomatrix(blk, distances, angles, levels, symmetric, normed)
   calc_prop = lambda blk : greycoprops(calc_glcm(blk), prop)

   # calculate properties of each block
   prop          = 'contrast'
   contrast      = dip.block_process(im, calc_prop, blk_shape)
   prop          = 'dissimilarity'
   dissimilarity = dip.block_process(im, calc_prop, blk_shape)
   prop          = 'homogeneity'
   homogeneity   = dip.block_process(im, calc_prop, blk_shape)
   prop          = 'energy'
   energy        = dip.block_process(im, calc_prop, blk_shape)
   prop          = 'correlation'
   correlation   = dip.block_process(im, calc_prop, blk_shape)
   prop          = 'ASM'
   ASM           = dip.block_process(im, calc_prop, blk_shape)

   return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

def get_probs(data, means, weights):
   # Function Description:
   #
   #   calculates a value between 0 and 1 that reflects the likelihood that a
   #   collection of pixels belong to a feature type; this likelihood is determined
   #   on a block-by-block basis via texture analysis; texture properties are
   #   calculated via GLCMs; the feature types are:
   #      - trees        -> land masses covered with trees
   #      - man_made     -> land masses with man-made structures
   #      - land_terrain -> land masses without trees or man-made structures
   #      - water_calm   -> water without waves/wakes
   #      - water_choppy -> water with waves/wakes
   #
   # Args:
   #
   #   data      : Description - collection of texture properties calculated from GLCMs
   #               Type        - ndarray
   #
   #   means     : Description - collection of mean values; each mean value corresponds
   #                             to a feature type and its represetative value for a
   #                             texture property
   #               Type        - list
   #
   #   weights   : Description - collection of weights used to put more/less emphasis
   #                             on each texture property
   #               Type        - list
   #
   # Returns:
   #
   #   ndarray containing 5 values for each block of the image; each of the 5 values
   #   correspond to one feature type and indicate the likelihood that the block is of
   #   that feature type

   # prop = number of properties
   # M    = rows of data for each property
   # N    = columns of data for each property
   # feat = number of features
   prop, M, N = data.shape
   prop, feat = means.shape

   probs = np.zeros((M, N, feat))

   for p in range(prop):
      ranks = np.zeros((M, N, feat))
      # iterate through each row of blocks
      for i in range(M):
         # iterate through each column of blocks
         for j in range(N):
            # iterate through each feature type
            for f in range(feat):
               ranks[i, j, f] = np.abs(means[p][f] - data[p][i, j]) * -1
            ranks[i, j] = rankdata(ranks[i, j]) * weights[p]
      probs += ranks # accumulate probabilities from each property

   probs /= feat*sum(weights) # normalize to 1

   return probs

def get_texture_mask(img, masked_feat, thresh):
   # Function Description:
   #
   #   generates mask for suppressing blocks that are believed to contain a given
   #   feature; the feature types are:
   #      - trees        -> land masses covered with trees
   #      - man_made     -> land masses with man-made structures
   #      - land_terrain -> land masses without trees or man-made structures
   #      - water_calm   -> water without waves/wakes
   #      - water_choppy -> water with waves/wakes
   #
   # Args:
   #
   #   img         : Description - input image
   #                 Type        - ndarray
   #
   #   masked_feat : Description - feature to be masked
   #                 Type        - string
   #
   #   thresh      : Description - threshold to compare probailities to (0.0 to 1.0)
   #                 Type        - float
   #
   # Returns:
   #
   #   ndarray of same size as input image with zeros for masked pixels and
   #   ones for non-masked pixels

   # set glcm parameters
   blk_sz    = 24
   distances = [2]
   angles    = [0]
   levels    = 256
   symmetric = True
   normed    = True

   # set rank weights
   properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
   weights    = [1, 1, 1, 1, 1, 1]

   # set texture property means
   features = ['trees', 'man_made', 'land_terrain', 'water_calm', 'water_choppy']
   means    = np.zeros((len(properties), len(features),))
   means[0] = [    74,    356,    168,     16,    232] # contrast
   means[1] = [   6.5,   11.1,    8.7,    2.9,   11.2] # dissimilarity
   means[2] = [0.1555, 0.1547, 0.1485, 0.3115, 0.0995] # homogeneity
   means[3] = [0.0526, 0.0528, 0.0507, 0.1220, 0.0405] # energy
   means[4] = [0.5709, 0.6902, 0.6407, 0.2388, 0.5736] # correlation
   means[5] = [0.0029, 0.0035, 0.0031, 0.0183, 0.0017] # asm

   # get image dimensions
   M,N = img.shape

   # calc GLCMs and get texture properties
   data = np.zeros((len(properties), int(M/blk_sz), int(N/blk_sz)))
   [data[0], # contrast
    data[1], # dissimilarity
    data[2], # homogeneity
    data[3], # energy
    data[4], # correlation
    data[5]  # asm
    ] = get_texture_properties(img, blk_sz, distances, angles, levels, symmetric, normed)

   # get probability of choppy water for each block
   probs = get_probs(data, means, weights)
   feat_prob = probs[:,:,features.index(masked_feat)]

   # generate mask from probabilities
   mask = np.zeros((M,N))
   R,C = feat_prob.shape
   for i in range(R):
      for j in range(C):
         if feat_prob[i,j] > thresh:
            mask[i*blk_sz:i*blk_sz+blk_sz, j*blk_sz:j*blk_sz+blk_sz] = 1

   return mask

#helper function used to generate the ground truth mask
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


#helper function used to generate an image from a collection of ground truth masks
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
    
def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks


def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

from skimage.morphology import label
def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]
    
    
def raw_prediction(c_img, c_img_nc):
    c_img_nc = np.expand_dims(c_img_nc, 0)/255.0
    cur_seg_nc = fullres_model.predict(c_img_nc)[0]
    
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, cur_seg_nc, c_img[0]

#############################
# Script entry point
#############################

if __name__ == '__main__':
    #Choose what type of run
    SimpleDemo = True
    
    if SimpleDemo:
        #will run on a single image
        ImageId = '0e0578e6b.jpg'
        Im_Dir = '../input/images/' #case of the selected example images
        display = False #set Boolean to display images
    else:
        #will run on a random subset of the testing directory
        ImageId = None
        Im_Dir = '../input/airbus-ship-detection/train_v2/' #case of the competition data set training directory
        Display = False #unset Boolean to display images
    
    num_test = 200 #set this variable to the number of images to be tested
    num_FAs = 0 #initialize variable to store number of false alarms generated
    num_FAs_unet = 0
    num_FAs_unet_clouds = 0
    num_success = 0 #initialize variable to store number of successful detections
    num_success_unet = 0
    num_success_unet_clouds = 0
    num_misses = 0 #initialize variable to store number of misses
    ships = 0
    no_ship = 0

    #generate a list of all files in the directory
    ship_dir = '../input/airbus-ship-detection'
    train_image_dir = os.path.join(ship_dir, 'train_v2')
    #read the .csv file containing the ground truth mask data
    masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        
    if not SimpleDemo:
        print("Compiling non-corrupt images")
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        # some files are too small/corrupt
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 
                                                                      os.stat(os.path.join(train_image_dir, 
                                                                                            c_img_id)).st_size/1024)
        unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50] # keep only +50kb files
        input_list = np.random.choice(unique_img_ids['ImageId'], size=num_test, replace = False)
        # Import the model from https://www.kaggle.com/hmendonca/u-net-model-with-submission

    else:
        input_list = [ImageId]

    
    from keras.models import load_model
    fullres_model = load_model('../input/u-net-model-with-submission/fullres_model.h5')
    
    print("Making predictions")
    for count, ImageId in enumerate(input_list):
        try:
            # print("Predicting image", count)
            #read in the image
            Input_image = dip.im_read(Im_Dir + ImageId.__str__() )
            
            #For the given Image get a run encoded representation of its mask
            mask_rle = masks.query('ImageId=="'+ImageId.__str__()+'"')['EncodedPixels']
            all_masks = masks_as_image(mask_rle)
            all_masks = all_masks[:,:,0]
            
            # filter clouds from image
            (Image, Cloud_mask) = remove_clouds(Input_image,k=0.141, D0=20, grayscale=True, cloud_mask_size=128)
            
            # Calculate the predictions from the U-net model
            first_seg, first_seg_cloud, first_img = raw_prediction(Input_image, Image)
            
            # U-net prediction with clouds suppressed
            u_net_pred_clouds = masks_as_color(multi_rle_encode(smooth(first_seg_cloud)[:, :, 0]))
            
            # U-net prediction with regular image
            u_net_pred = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
            
            # convert image to grayscale
            img = dip.rgb2gray(Image)
        
            # get man-made mask
            man_made_mask = get_texture_mask(img, 'man_made', 0.6)
            
            Image = dip.rgb2ycbcr(Image) #convert to YCbCr color space
            Image = Image[:,:,0] #keep only luminance channel
        
            Image = cv2.medianBlur(Image.astype(np.float32),5) #Sharpen the image with a median filter for better edge detection
        
            ####### Edge Image Generation ############
            Edge_image = dip.transforms.edge_detect(Image.astype(float),'scharr') # use Scharr operator (rotation invariant) to calculate gradient magnitude in the image
            Edge_image = Edge_image.astype(np.float)/Edge_image.max().astype(np.float) #Normalize the edge gradient magnitude between [0,1]
            Edge_image = np.power(Edge_image,1) #Element-wise squaring of the edge gradient magnitude
        
            ###### Mixed Image Genreation ############
            Mixed_image = (Image/Image.max()*255./2.+Edge_image/Edge_image.max()*255/2.) #Sharpening Operation which intensifies the edges in the image
            Mixed_image = (Mixed_image/(Mixed_image.max().astype(np.float))*255.).astype(np.uint8) #normalize the image back to [0,255]
        
            ###### Image Segmentation ###############
            #Use Otsu's method to attempt to separate the background class from the foreground class
            cv_k,Segmented_Image = cv2.threshold(Mixed_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
            ###### Hole Filling ###########
        
            close_kernel = np.ones((3,3),np.uint8) #kernel used for filling holes
            Closed_image = cv2.morphologyEx(Segmented_Image,cv2.MORPH_CLOSE, close_kernel) #Apply morphological operator to close holes in the image
        
            ##### Denoising #######
            open_kernel = np.ones((4,4),np.uint8) #kernel used for removing noise
            Denoised_image = cv2.morphologyEx(Closed_image,cv2.MORPH_OPEN, open_kernel)
            
            im2, contours, hierarchy = cv2.findContours(Denoised_image.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            BB_mask = np.zeros(im2.shape) #this image will store the mask for the ships
            for i,cnt in enumerate(contours):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                dim1 = np.linalg.norm(box[0]-box[1])
                dim2 = np.linalg.norm(box[1]-box[2])
        
                #per "Ship Detection From Optical Satellite Images Based on Sea Surface Analysis"
                #the Length-width ratio and Compactness shape features are calcualted as:
                LW_ratio = max(dim1/dim2, dim2/dim1) #by defintion length will be the longer dimension of rectangle
                Compactness = np.power( (2.*dim1+2.*dim2), 2)/(dim1*dim2)
                #The following are the paper's recommended limits on each feature
                LW_max =  16.5
                LW_min =  3.5
                Comp_max = 58
                Comp_min = 15
                if LW_ratio > LW_min and LW_max > LW_ratio:
                    if Compactness>Comp_min and Comp_max > Compactness:
                        cv2.fillPoly(BB_mask, [box], 1) #only accept mask for potential ships
            
            #Mask the image with the acceptable regions after shape analysis
            Shaped_image = BB_mask*Denoised_image
            #Remove any false alarms generated by choppy water/land/clouds by multiplying masks together
            Final_mask = BB_mask*man_made_mask*np.logical_not(Cloud_mask)
            #generate the final segmented image by multiplying the denoised image with the mask
            Feature_image = Final_mask*Denoised_image
            
            if display == True:
                
                dip.figure(figsize=(10,10))
                ax = dip.subplot(3,5,1)
                dip.imshow(Input_image)
                ax.set_title("Input Image")
                ax = dip.subplot(3,5,2)
                dip.imshow(Edge_image, 'gray')
                ax.set_title("Edge Image")
                ax = dip.subplot(3,5,3)
                dip.imshow(Mixed_image, 'gray')
                ax.set_title("Mixed Image")
                ax = dip.subplot(3,5,4)
                dip.imshow(Segmented_Image*255, 'gray') #multiply by 255 to get binary image for display
                ax.set_title("Segmented Image")
                ax = dip.subplot(3,5,5)
                dip.imshow(Denoised_image*255, 'gray')
                ax.set_title("Denoised Image")
                ax = dip.subplot(3,5,6)
                dip.imshow(Final_mask*255, 'gray')
                ax.set_title("Final Mask")
                ax = dip.subplot(3,5,7)
                dip.imshow(Shaped_image*255, 'gray')
                ax.set_title("Shape Analysis Results")
                ax = dip.subplot(3,5,8)
                dip.imshow(Feature_image*255, 'gray')
                ax.set_title("Feature Image")
                ax = dip.subplot(3,5,9)
                dip.imshow(u_net_pred*255, 'gray')
                ax.set_title("U-net Pred")
                ax = dip.subplot(3,5,10)
                dip.imshow(u_net_pred_clouds*255, 'gray')
                ax.set_title("U-net + Clouds")
                ax = dip.subplot(3,1,3)
                dip.imshow(all_masks*255, 'gray')
                ax.set_title("Ground Truth")
                dip.show()
        
            # No ship present
            if not np.any(all_masks):
                no_ship+=1
                if np.any(Final_mask):
                    num_FAs+=1 #False alarm is claiming ship pixels exist but no ship is in the image
                else:
                    num_success+=1
                if np.any(u_net_pred):
                    num_FAs_unet+=1
                else:
                    num_success_unet+=1
                if np.any(u_net_pred_clouds):
                    num_FAs_unet_clouds+=1
                else:
                    num_success_unet_clouds+=1

            # Ship present
            else:
                ships+=1
                if np.any(all_masks*Final_mask):
                    num_success+=1
                if np.any(all_masks*u_net_pred):
                    num_success_unet+=1
                if np.any(all_masks*u_net_pred_clouds):
                    num_success_unet_clouds+=1
        except:
            print("Prediction",count,"failed... skipping image")
            pass
        
    print("Combo success:", num_success / (ships +  no_ship))
    print("U-net success:", num_success_unet / (ships + no_ship))
    print("U-net + Clouds success", num_success_unet_clouds / (ships + no_ship))
    print(ships + no_ship, "total images analyzed")
    try:
        print("Combo FAR:", num_FAs / no_ship)
        print("U-net FAR:", num_FAs_unet / no_ship)
        print("U-net + Clouds FAR:", num_FAs_unet_clouds / no_ship)
        print(no_ship, "no-ship images")
    except:
        print("Only image(s) with ships were analyzed, no False Alarm Rate to report")
        pass