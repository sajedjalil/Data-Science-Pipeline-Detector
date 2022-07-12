# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


print(len(train_images))

train_images = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_images)

test_images = test_images[:]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS,CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        
        data[i] = image/255
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_images)


labels = []
for i in train_images:
    if 'dog' in i:
        labels.append([1,0])
    else:
        labels.append([0,1])



from keras.layers import Dense, Dropout, Activation,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.models import Model
from keras import optimizers
import numpy as np


from keras import regularizers

inputs = Input(shape=(ROWS, COLS,CHANNELS))

x = Flatten()(inputs)
x = Dense(512,activation='relu')(x)
x = Dropout(0.8)(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.8)(x)
predictions = Dense(2,activation='softmax')(x)


model = Model(inputs=inputs,outputs=predictions)
model.compile(optimizer=optimizers.Adam(lr=1e-4,decay=1e-5),loss='binary_crossentropy',metrics=['accuracy'])
loss_ = model.fit(train, labels,
          epochs=40,
          batch_size=30)
print(loss_)

predicts = model.predict(train)
y_pre =  np.array(predicts)[:,0]
y_true = np.array(labels)[:,0]

from sklearn.metrics import log_loss
log_loss_ = log_loss(y_true,y_pre)
print(log_loss_)

import math
predicts = []
batch_size = 100
for i in range(0,math.ceil(len(test_images)/batch_size)):
    batch_test =  test_images[i*batch_size:(i+1)*batch_size]
    batch_test = prep_data(batch_test)
    predict_test = model.predict(batch_test)
    predict_test = predict_test[:,0]
    predicts = np.concatenate((predicts,predict_test),axis=0)

dataframe = {
    'id':range(1,len(predicts)+1),
    'label':predicts
    }
    
dataframe = pd.DataFrame(dataframe)
dataframe.to_csv('out.csv',index=False)