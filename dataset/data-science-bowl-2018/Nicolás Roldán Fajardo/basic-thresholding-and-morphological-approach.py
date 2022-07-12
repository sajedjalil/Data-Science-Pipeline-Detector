import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import scipy
import scipy.ndimage as snd

from scipy import ndimage
from skimage import img_as_float
from skimage.io import imread
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage.filters import threshold_local
from skimage.filters.thresholding import threshold_otsu



# Input data files are available in the "../input/" directory.
DATA = '../input/stage1_test'


def yield_images():
    for p in glob.glob(os.path.join(DATA, '*', 'images', '*')):
        yield p




def dark_back_plane(I):
    maximus = np.array([])
    for i in range(0,3):
        plane = I[:,:,i]
        hist,bins = np.histogram(plane.ravel(),256,[0,255])
        maximus = np.append( maximus,[bins[np.argmax(hist)]])#intervalo de frecuencias donde ocurre el maximo
    plane = I[:,:,np.argmax(maximus)]#plano con más información
    if max(maximus) > 128:#intervalo de frecuencias correspondiente al máximo entre los histogramas
        plane = invert(plane)
    return plane
    
def dark_back(I):
    skew = np.array([])
    cont = 0
    for i in range(0,3):
        plane = I[:,:,i]
        hist,bins = np.histogram(plane.ravel(),256,[0,255])
        skew = scipy.stats.skew(plane.ravel())
        if skew < 0:
            cont+=1    
    if cont >= 2:
        plane = invert(rgb2gray(I))
    return plane
    
        
def get_image(p):
    img_id = p.split('/')[-1][:-4]
    img = imread(p)
    img = img_as_float(img)
    img_rgb = img if img.shape[2] == 3 else rgba2rgb(img)
    img_dark = dark_back(img_rgb)
    return img_dark,img_id

def otsu(m_I):
    thresh = threshold_otsu(m_I)#Threshold estimation
    BW = m_I > thresh
    return BW

# In[] Segmentation by adaptive thresholding (regions)
def adaptive_thresh(img):
    block_size = 25
    thresh = threshold_local(img,block_size,method='gaussian',offset=-1,param=5)
    b = img > thresh
    return b

def morpho_op(BW):
    s = [[0,1,0],[1,1,1],[0,1,0]]#structuring element (diamond shaped)
    m_morfo = snd.morphology.binary_opening(BW,structure=s,iterations=1)
    m_morfo = snd.morphology.binary_closing(m_morfo,structure=s,iterations=1)
    M_filled = snd.morphology.binary_fill_holes(m_morfo,structure=s)
    return M_filled

def rle_encode(img):
    # Ref. https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    pixels = img.flatten('F')
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

def submission(f):
    for i, p in enumerate(yield_images()):
        img_dark, img_id = get_image(p)
        img_mask = adaptive_thresh(img_dark)
        BW = morpho_op(img_mask)
        print(i,img_id)
        labeled_array, num_features = snd.label(BW)
        for label in range(1, num_features+1):
            mask = labeled_array == label
            mask_encoded = rle_encode(mask)
            print(img_id, mask_encoded, sep=',', file=f)


if __name__ == '__main__':
    with open('submission_08Apr2018.csv', 'w') as f:
        print("ImageId,EncodedPixels", file=f)
        submission(f)