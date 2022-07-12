# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python import keras as keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def load_data(batch_size=16, mode='categorical'):
    dataframe = pd.read_csv('../input/train.csv')
    dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)
    gen = ImageDataGenerator(rescale=1. / 255., validation_split=0.1, horizontal_flip=True, vertical_flip=True)

    _train = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id',
                                     y_col='has_cactus', has_ext=True, target_size=(32, 32),
                                     class_mode=mode, batch_size=batch_size, shuffle=True, subset='training')
    _test = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id',
                                    y_col='has_cactus', has_ext=True, target_size=(32, 32),
                                    class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    return _train, _test

def load_eval_data(batch_size=16, mode='categorical'):
    dataframe = pd.read_csv('../input/sample_submission.csv')
    dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)
    gen = ImageDataGenerator(rescale=1. / 255., horizontal_flip=True, vertical_flip=True)

    _eval = gen.flow_from_dataframe(dataframe, directory='../input/test/test', x_col='id',
                                    y_col='has_cactus', has_ext=True, target_size=(32, 32),
                                    class_mode=mode, batch_size=batch_size, shuffle=False)
    return _eval

def model():
    shape = (32, 32, 3)
    model_layers = [
        keras.layers.Conv2D(8, (2, 2), activation='relu', padding='same', name='block1_conv1', input_shape=shape),
        keras.layers.Dropout(0.01),
        keras.layers.Conv2D(8, (4, 4), activation='relu', padding='same', name='block1_conv2'),
        keras.layers.Dropout(0.01),
        keras.layers.Conv2D(8, (8, 8), activation='relu', padding='same', name='block1_conv3'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),

        keras.layers.Conv2D(32, (4, 4), activation='relu', padding='same', name='block2_conv1'),
        keras.layers.Dropout(0.05),
        keras.layers.Conv2D(32, (8, 8), activation='relu', padding='same', name='block2_conv2'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),

        keras.layers.Conv2D(64, (8, 8), activation='relu', padding='same', name='block3_conv1'),
        keras.layers.Conv2D(64, (8, 8), activation='relu', padding='same', name='block3_conv2'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),

        keras.layers.Conv2D(128, (8, 8), activation='relu', padding='same', name='block4_conv1'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu', name='fc1'),
        keras.layers.Dense(2, activation='sigmoid', name='predictions')
    ]

    result = keras.Sequential(model_layers, name='aci')
    result.compile(optimizer=keras.optimizers.SGD(lr=1e-2), loss=keras.losses.binary_crossentropy,
                   metrics=['accuracy'])
    result.summary()
    return result

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-3)
train, test = load_data()
model = model()
model.fit_generator(train, epochs=70, validation_data=test, shuffle=False, callbacks=[reduce_lr])
eval = load_eval_data()
predictions = model.predict_generator(eval)
dataframe = pd.read_csv('../input/sample_submission.csv')
dataframe['has_cactus'] = np.array(predictions[:, 0] < predictions[:, 1]).astype(np.float64)
dataframe.to_csv('sample_submission.csv', index=False)
