# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:52:56 2019

@author: kalpa-user
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import keras
from keras.models import Sequential #add layers one at a time until we are happy with our network architecture.
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D #Fully connected layers are defined using the Dense class.
from keras.layers.normalization import BatchNormalization

TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test'
IMG_SIZE = 224

# =============================================================================
# FUNCTIONS TO PREPARE DATASET FOR CNN
# =============================================================================
def label_img(img):
    # ex. dog.93.png -> [-1] = png; [-2] = 93; [-3] = dog;
    if len(img.split('.')) != 3:
        return 'error'
    else:
        word_label = img.split('.')[-3]
        if word_label == 'cat':
            return [1,0]
        elif word_label == 'dog':
            return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if label_img(img) == 'error':
            continue
        else:
            label = label_img(img)
            full_path = os.path.join(TRAIN_DIR, img)
            grayscale_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            resized_grayscale_img = cv2.resize(grayscale_img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(resized_grayscale_img), np.array(label)])
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    #print (training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_id = img.split('.')[0]
        img_grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_grayscale_resize = cv2.resize(img_grayscale, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img_grayscale_resize), img_id])
    
    np.save('test_data.npy', testing_data)
    return testing_data

# =============================================================================
# CREATE DATASET
# =============================================================================
train_data = create_train_data()
#or if u have already train data set
#train_data = np.load('../input/224-trainingset1/training_data.npy', allow_pickle = True)

# =============================================================================
# BUILDING THE CONVNET
# =============================================================================
#create model
model = Sequential()

#--------------------------------------------
#add model layers
#convnet.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)))
#convnet.add(Conv2D(64, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(32, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(64, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(32, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(64, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(32, kernel_size=2, activation='relu'))
#convnet.add(Conv2D(64, kernel_size=2, activation='relu'))
#convnet.add(Flatten())
#convnet.add(Dense(1024, activation='relu'))
#convnet.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
#convnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#--------------------------------------------

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(IMG_SIZE,IMG_SIZE,1), kernel_size=(11,11), activation='relu', strides=(4,4), padding='valid'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='valid'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, activation='relu', input_shape=(IMG_SIZE*IMG_SIZE*1,)))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096, activation='relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000, activation='relu'))
# Add Dropout
model.add(Dropout(0.4))
# Output Layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# =============================================================================
# VALIDATION DATASET
# =============================================================================
train = train_data[:-500]
validation = train_data[-500:]

train_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #pixel data
train_Y = np.array([i[1] for i in train])
#print (train_Y)

validation_X = np.array([i[0] for i in validation]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #pixel data
validation_Y = np.array([i[1] for i in validation])
#print (validation_Y)

# =============================================================================
# TRAIN THE NETWORK
# =============================================================================
#train the model
model.fit(train_X, train_Y, validation_data=(validation_X, validation_Y), epochs=10)

model.save('alexnet.h5')  # creates a HDF5 file 'my_model.h5'

from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server