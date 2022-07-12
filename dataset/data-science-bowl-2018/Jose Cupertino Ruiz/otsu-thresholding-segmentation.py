# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

from scipy import ndimage
from skimage import img_as_float
from skimage import io
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_bilateral
from skimage.util import invert


import glob
import numpy as np
import os


# Input data files are available in the "../input/" directory.
DATA = '../input/stage1_test'


def yield_images():
    """ Iterate over test images
    """
    for p in glob.glob(os.path.join(DATA, '*', 'images', '*')):
        yield p


def dark_image(img):
    """ Ensures dark background
    """
    img_float = img_as_float(img)
    img_dark = img_float if np.mean(img_float) < 0.5 else invert(img_float)
    return img_dark
    
        
def get_image(p):
    """ Receives the path a the image
        Perform bilateral denoising
        Return image array and its ID
        Convert to Gray scale
    """
    img_id = p.split('/')[-1][:-4]
    img = io.imread(p)
    img_rgb = img if img.shape[2] == 3 else rgba2rgb(img)
    img_dark = dark_image(img_rgb)
    img_denoise = denoise_bilateral(img_dark, multichannel=True)
    img_gray = rgb2gray(img_denoise)
    return img_gray, img_id


def segmentation(img):
    """ Apply Otsu thresholding
        http://www.scipy-lectures.org/packages/scikit-image/#image-segmentation
    """
    val = threshold_otsu(img)
    mask = img > val
    return mask


def rle_encode(img):
    """ Ref. https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    """
    pixels = img.flatten('F')
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def submission(f):
    """ Loop over images
    """
    for i, p in enumerate(yield_images()):
        img_gray, img_id = get_image(p)
        img_mask = segmentation(img_gray)
        print(i, img_id)
        labeled_array, num_features = ndimage.label(img_mask)
        """ Loop over nuclei
        """
        for label in range(1, num_features+1):
            mask = labeled_array == label
            mask_encoded = rle_encode(mask)
            print(img_id, mask_encoded, sep=',', file=f)


if __name__ == '__main__':
    with open('submission_07Feb2018.csv', 'w') as f:
        print("ImageId,EncodedPixels", file=f)
        submission(f)