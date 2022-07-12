"""

This kernel tries to fix the colors of the JPG

Created on Tue May  9 10:04:30 2017

@author: Paulo Pinto

"""

import glob
from tqdm import tqdm
import spectral as sp
from skimage import io

def correct_jpg(in_path):
    path_jpg = glob.glob(in_path + '*.jpg')
    for path in tqdm(path_jpg, miniters=20):
        jpg = io.imread(path)
        jpg = jpg[:,:,:3]
        sp.save_rgb(path, jpg, colors=sp.spy_colors)

#Uncomment the lines below to run.
#correct_jpg('../input/train-jpg/')
#correct_jpg('../input/test-jpg/')

