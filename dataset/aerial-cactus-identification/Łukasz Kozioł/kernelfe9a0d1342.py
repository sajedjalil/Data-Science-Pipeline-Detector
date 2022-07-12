# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications.vgg16 import VGG16

#from __future__ import division

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df['has_cactus'] = df['has_cactus'].astype('bool').astype('str')

datagen = ImageDataGenerator(rescale=1./255)
# datagen = ImageDataGenerator(rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

train_generator = datagen.flow_from_dataframe(dataframe=df, directory="../input/train/train", x_col="id", y_col="has_cactus", class_mode="binary", target_size=(32,32), batch_size=175, shuffle=False)

def get_model_from_egyptian_kernel():
    model = Sequential()
    model.add( Conv2D(64, (5, 5), input_shape=(32, 32, 3)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(64, (5, 5)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(128, (5, 5)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(128, (5, 5)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(256, (3, 3)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(256, (3, 3)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Conv2D(512, (3, 3)))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Flatten())
    
    
    model.add( Dense(100))
    model.add( BatchNormalization())
    model.add( LeakyReLU(alpha=0.3))
    
    model.add( Dense(1, activation='sigmoid'))
    return model

input_shape = (32, 32, 3)

model = Sequential([
    Conv2D(64, (3,3), input_shape=input_shape),
    MaxPool2D((2, 2)),
    
    Conv2D(128, (3,3)),
    MaxPool2D((2, 2)),
    
    Conv2D(256, (3,3)),
    MaxPool2D((2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(.5),
    Dense(1, activation='sigmoid')    
])

#model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# for layer in base_model.layers:
#     layer.trainable = False

# model = Sequential([
#     base_model,
    
#     Flatten(), #<= bridge between conv layers and full connected layers
    
#     Dense(1024, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model from Egypian kernel https://www.kaggle.com/mariammohamed/simple-cnn
#model = get_model_from_egyptian_kernel()
#opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2)
early_stop = EarlyStopping(monitor='loss', patience=3)

#model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 8, epochs=30, callbacks=[lr_sched, early_stop])

history = model.fit_generator(
   train_generator,
   steps_per_epoch=50,
   epochs=20,
   #callbacks=[lr_sched, early_stop],
   #validation_steps=50,
   verbose=2)

df_test_final = pd.DataFrame({"id": os.listdir("../input/test/test")})
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=df_test_final, directory="../input/test/test", class_mode=None, x_col="id" , target_size=(32, 32), batch_size=40, shuffle=False)

predictions = model.predict_generator(test_generator, steps=100)
df_test_final['has_cactus'] = predictions
df_test_final.to_csv('test_predicion.csv', index=False)