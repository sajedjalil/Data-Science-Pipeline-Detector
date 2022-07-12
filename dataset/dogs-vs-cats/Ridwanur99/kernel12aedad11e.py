# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:37:46 2020

@author: ridwanur99
"""
#importing the important packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#initialising the cnn
classifier = Sequential()
#Step-1 :- convolutionfilters
classifier.add(Convolution2D(filters = 32, kernel_size = 3, input_shape = (64,64,3), activation='relu'))
#step2:- pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Step-3 flattening
classifier.add(Flatten())
#step-4:- Full Connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/train1',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) 