# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
FILE_TRAIN = '/kaggle/input/Kannada-MNIST/train.csv'
FILE_TEST = '/kaggle/input/Kannada-MNIST/test.csv'

def get_cropped_image(image):
    left = 0
    while image[left, :].sum() == 0:
        left += 1
    right = 27
    while image[right, :].sum() == 0:
        right -= 1
    up = 0
    while image[:, up].sum() == 0:
        up += 1
    down = 27
    while image[:, down].sum() == 0:
        down -= 1
    trimmed = Image.fromarray(np.uint8(image[left:right + 1, up:down + 1]))
    return np.array(trimmed.resize((28, 28)))

channels = 1
width, height = 28, 28

data = pd.read_csv(FILE_TRAIN)
train_cases = data.shape[0]

prepared_data = []
for i in range(train_cases):
    row = data.iloc[i]
    arr = np.array(row[1:]).reshape((28, 28))
    #prepared_data.append(get_cropped_image(arr))
    prepared_data.append(arr)

prepared_data = np.array(prepared_data)
prepared_target = np.array(data['label']).reshape((-1, 1))

features_train = prepared_data.reshape((prepared_data.shape[0], width, height, channels))
target_train = np_utils.to_categorical(prepared_target)

validation_rate = 0.2
validation_threshold = int(features_train.shape[0] * (1 - validation_rate))

features_test = features_train[validation_threshold:, ] / 255
target_test = target_train[validation_threshold:, ]

features_train = features_train[:validation_threshold, ] / 255
target_train = target_train[:validation_threshold, ]

number_of_classes = target_train.shape[1]

imagegen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=4,
    height_shift_range=4,
    zoom_range=0.1
)

nets = []
for i in range(3):
    print('network', i + 1)
    network_name = 'model_%d.h5' % i

    print('training..')
    network = Sequential()

    network.add(Conv2D(
        filters=32,
        kernel_size=3,
        input_shape=(width, height, channels),
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=32,
        kernel_size=5,
        padding='same',
        strides=2,
        activation='relu'
    ))
    
    network.add(BatchNormalization())
    network.add(Dropout(0.4))

    network.add(Conv2D(
        filters=48,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=48,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=48,
        kernel_size=5,
        padding='same',
        strides=2,
        activation='relu'
    ))

    network.add(BatchNormalization())
    network.add(Dropout(0.4))

    network.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    network.add(BatchNormalization())
    network.add(Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        strides=2,
        activation='relu'
    ))

    network.add(BatchNormalization())
    network.add(Dropout(0.4))

    network.add(Flatten())
    network.add(Dense(units=128, activation='relu'))
    network.add(BatchNormalization())
    network.add(Dropout(0.4))

    network.add(Dense(units=number_of_classes, activation='softmax'))

    network.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    network.fit_generator(
        imagegen.flow(features_train, target_train, batch_size=32),
        epochs=30,
        verbose=2,
        steps_per_epoch=features_train.shape[0] // 32,
#         callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        validation_data=(features_test, target_test)
    )
    nets.append(network)

data = pd.read_csv(FILE_TEST)
test_cases = data.shape[0]

prepared_data = []
for i in range(test_cases):
    row = data.iloc[i]
    arr = np.array(row[1:]).reshape((28, 28))
    #prepared_data.append(get_cropped_image(arr))
    prepared_data.append(arr)

prepared_data = np.array(prepared_data) / 255

features_test = prepared_data.reshape((prepared_data.shape[0], width, height, channels))

print('predicting..')
prediction = np.zeros((features_test.shape[0], 10))
for i in range(len(nets)):
    print(i, end=' ')
    prediction += nets[i].predict(features_test)

classes = []
for i in prediction:
    classes.append(np.argmax(i))

print('\nsaving..')
df = pd.DataFrame()
df["id"] = [i for i in range(test_cases)]
df["label"] = classes

df.to_csv('submission.csv', index=False)