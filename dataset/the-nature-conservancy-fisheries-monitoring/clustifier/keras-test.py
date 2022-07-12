# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

print (keras.__version__)

import h5py
import os

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#path to training data
DATA_PATH = '../input/train'

#Number of clusters for K-Means
N_CLUSTS = 5#250

#Number of clusters used for validation
N_VAL_CLUSTS = 1#50

SEED = 42
np.random.seed(SEED)

##############################################
#######NORMALIZED IMAGE SIZE
##############################################
IMG_WIDTH = 640
IMG_HEIGHT = 360

##############################################
#######SUBSAMPLE DATA
##############################################

#how many images to take?
SAMP_SIZE = 8

subsample = []
for fish in os.listdir(DATA_PATH):
    if(os.path.isfile(os.path.join(DATA_PATH, fish))): 
        continue
    subsample_class = [os.path.join(DATA_PATH, fish, fn) for 
                       fn in os.listdir(os.path.join(DATA_PATH, fish))]
    subsample += subsample_class
subsample = subsample[:SAMP_SIZE]

base_model = VGG16(weights = None, include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
#base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
model = Model(input = base_model.input, output = base_model.get_layer('block4_pool').output)

# Any results you write to the current directory are saved as output.