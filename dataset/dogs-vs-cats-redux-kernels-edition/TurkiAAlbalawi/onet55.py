
import numpy as np 
import pandas as pd 

from subprocess import check_output

import os, cv2, random
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 32
COLS = 32
CHANNELS = 1

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] 


test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        
    
    return data

train = prep_data(train_images)
test = prep_data(test_images)



labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
        
train = train.reshape(-1, 32,32,1)
test = test.reshape(-1, 32,32,1)
X_train = train.astype('float32')
X_test = test.astype('float32')
X_train /= 255
X_test /= 255
Y_train=labels

x=np.array(X_train)
print(x.shape)
X_valid = X_train[:5000,:,:,:]
Y_valid =   Y_train[:5000]
X_train = X_train[5001:25000,:,:,:]
Y_train  = Y_train[5001:25000]