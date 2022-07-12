import os

import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import numpy as np
from skimage import io

PLANET_KAGGLE_ROOT = os.path.abspath("../input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

def calibrate_image(rgb_image):
    # Transform test image to 32-bit floats to avoid 
    # surprises when doing arithmetic with it 
    calibrated_img = rgb_image.copy().astype('float32')

    # Loop over RGB
    for i in range(3):
        # Subtract mean 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])
        # Normalize variance
        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])
        # Scale to reference 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]
        # Clip any values going out of the valid range
        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

    # Convert to 8-bit unsigned int
    return calibrated_img.astype('uint8')

# Pull a list of 20000 image names
jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]
# Select a random sample of 100 among those
np.random.shuffle(jpg_list)
jpg_list = jpg_list[:100]

ref_colors = [[],[],[]]
for _file in jpg_list:
    # keep only the first 3 bands, RGB
    _img = image.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]
    # Flatten 2-D to 1-D
    _data = _img.reshape((-1,3))
    # Dump pixel values to aggregation buckets
    for i in range(3): 
        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()
    
ref_colors = np.array(ref_colors)

ref_means = [np.mean(ref_colors[i]) for i in range(3)]
ref_stds = [np.std(ref_colors[i]) for i in range(3)]

def load_img(image_name, s):
    if 'train' in image_name:
        path = '../input/train-tif-v2/{}.tif'.format(image_name)
    elif 'file' in image_name:
        path = '../input/test-tif-v2/{}.tif'.format(image_name)
    else:
        path = '../input/test-tif-v2/{}.tif'.format(image_name)
    bgr = io.imread(path)
    if bgr is not None:
        return calibrate_image(bgr)
    else:
        return np.zeros((s, s, 3))


mosaic_idx = [
    [None, None, None, None, None],
    [None, None, 'train_17984', None, None],
    [None, 'train_19637', 'file_4712', 'train_28191', None],
    [None, None, 'train_40193', None, None],
    [None, None, None, None, None],
]

s = 256
mosaic = np.zeros((s*5, s*5, 3), dtype=np.uint8)

for j in range(5):
    for i in range(5):
        if mosaic_idx[j][i]:
            mosaic[j*s:s*(j+1), i*s:(i+1)*s, :] = load_img(mosaic_idx[j][i], s)[:,:,:3]

image.imsave('out.png', mosaic)