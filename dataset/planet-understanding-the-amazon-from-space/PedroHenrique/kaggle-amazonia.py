import theano
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.wrappers.scikit_learn import KerasClassifier

import cv2
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

theano.config.openmp = True

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from keras import backend as K

def fbeta(y_true, y_pred, threshold_shift=0):
        beta = 2

    # just in case of hipster activation at the final layer
        y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
        y_pred_bin = K.round(y_pred + threshold_shift)

        tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
        fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        beta_squared = beta ** 2
        return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

x_train = []
x_test = []
y_train = []

seed = 7

np.random.seed(seed)

df_train = pd.read_csv('../input/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

x_resize = 32
y_resize = 32

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (x_resize, y_resize), interpolation = cv2.INTER_AREA))
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

def create_model():

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(x_resize, y_resize, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(20, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(17, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=[fbeta])
        return model

model = create_model()

model.fit(x_train, y_train, batch_size=128, epochs=4,
          verbose=1,
          validation_data=(x_valid, y_valid))
from sklearn.metrics import fbeta_score

y_train_img = []

for p in tqdm(range(0, 40669)):
        str_obj = '../input/test-jpg-v2/test_' + str(p) + '.jpg'
        img1 = cv2.imread(str_obj)
        y_train_img.append(cv2.resize(img1, (x_resize, y_resize)))
'''
for p in tqdm(range(0, 20522)):
        str_obj = '../input/test-jpg-additional/file_' + str(p) + '.jpg'
        img1 = cv2.imread(str_obj)
        y_train_img.append(cv2.resize(img1, (x_resize, y_resize)))
'''
y_train_img = np.array(y_train_img, np.float32) / 255

p_valid = model.predict(y_train_img,verbose=2)
print(y_valid)
print(p_valid)

file = open ("output.csv", "w")
p = 'image_name,tags' + '\n'
file.write(p)
for i in range(0,40669):
        p = 'test_' + str(i) + ','
        for y in range(0, 17):
                if(p_valid[i][y]>0.8):
                        p += inv_label_map[y]+' '
        print(p)
        p += '\n'
        file.write(p)

for i in range(0,20522):
        p = 'file_' + str(i) + ','
        for y in range(0, 17):
                if(p_valid[i][y]>0.8):
                        p += inv_label_map[y]+' '
        print(p)
        p += '\n'
        file.write(p)
