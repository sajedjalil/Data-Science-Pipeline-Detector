import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import cv2
from tqdm import tqdm

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('../input/train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values[:20000], miniters=1000):
    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 15000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
              

batch_size = 128
gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    zca_whitening=False,
    zoom_range=[1.0, 1.2],
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
).flow(x_train, y_train, batch_size=batch_size)

model.fit_generator(
    gen,
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=5,
    verbose=1,
    validation_data=(x_valid, y_valid)
)
          
from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > 0.5, beta=2, average='macro'))
