# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv).
import os
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
from keras.preprocessing import image   # for preprocessing the images
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
import glob
import tqdm
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import glob
import pandas
import cv2
from sklearn.utils import shuffle
import keras
from sklearn.model_selection import train_test_split

#kaggle/input/Deepfake Detection Challenge/train_sample_videos
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

files = glob.glob("/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4")
#map = open("truesight_map.csv","r")
X = []
Y = []
files1 = []
map = pandas.read_csv("/kaggle/input/deepcsv/deepfake.csv")
#files = shuffle(files, random_state=20)
for i in files:
    for j in map.iterrows():
        row = j[1]
        if "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"+row[0] == i:
            files1.append(i)
            Y.append(row["label"])
def video3d(filename,height,width,depth, color=False, skip=True):
    cap = cv2.VideoCapture(filename)
    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if skip:
        frames = [x * nframe / depth for x in range(depth)]
    else:
        frames = [x for x in range(depth)]
    framearray = []

    for i in range(depth):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        if type(frame) == np.ndarray:
            frame = cv2.resize(frame, (height, width))
            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            pass


    cap.release()
    return np.array(framearray)
    return(framearray)

#print(video3d(files[0],240,240,400

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(64,64,15,1),data_format='channels_last'))
model.add(Activation('relu'))
model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
model.add(Dropout(0.25))
model.add(Conv3D(128, padding="same", kernel_size=(3, 3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam", metrics=["accuracy"])
print(model.summary())
fog = 0
y = []
for i in Y:
    if i == "FAKE":
        y.append(0)
    else:
        y.append(1)
for i in range(400):
    f = []
    a = video3d("/kaggle/input/deepfake-detection-challenge/train_sample_videos/"+map["_key"][i],64,64,15)
    X.append(a)
    print(i)
X = np.ndarray((len(X),64,64,15,1))
y1 = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y1, random_state=42, test_size=0.2, stratify = y1)
model.fit(X_train, y_train, epochs=10, verbose=1, batch_size=100)
#model.evaluate(X_test,y_test)
keras.backend.clear_session()
'''
predic = pandas.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

d = []
for i in range(10):
    a = video3d("/kaggle/input/deepfake-detection-challenge/test_videos/"+predic["filename"][i],64,64,15)
    a = np.array(a)
    d.append(a)
d = np.array(d)
d = d.reshape((len(d),64,64,15,1))
model.predict(d)
#f = np.array(f)
#f = f.reshape((len(X),64,64,15,1))
#print(type(f))
'''