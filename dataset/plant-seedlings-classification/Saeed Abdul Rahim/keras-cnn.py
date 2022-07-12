# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 16:49:42 2017

@author: sae_x

---Directory path changed--- 25% split for validation set
plant/
    train/
        category1/
        category2/...
    validation/
        category1/
        category2/...
    test/
        test/
"""
# importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# for neural network
classifier = Sequential()
# adding feature extract or filter layer with rectifier function
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# adding layer for pooling
classifier.add(MaxPooling2D(2, 2))
# Repeating for more accuracy
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(2, 2))
# Flattening Step for putting in neural network
classifier.add(Flatten())
# Hidden layer for neural network with rectifier function
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# output layer with softmax fuction for multiple outputs
classifier.add(Dense(output_dim = 12, activation = 'softmax'))
# neural network with stochastic gradient descent abd categorical crossentropy for multiple outputs
classifier.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"])
# augmenting the image
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
# using flow_from_directory method for generator to classify using directories
train_generator = train_datagen.flow_from_directory(
        'plant/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'plant/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
# Fit
classifier.fit_generator(
        train_generator,
        steps_per_epoch=3558,
        epochs=8,
        validation_data=validation_generator,
        validation_steps=1192)

test_generator = test_datagen.flow_from_directory(
        'plant/test',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical')

predict = classifier.predict_generator(test_generator, len(test_generator.filenames))