# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1712)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from scipy.misc import imread, imresize
from keras.layers.advanced_activations import ELU




def create_model_v1(img_rows, img_cols, color_type=1):
    nb_classes = 10
    # number of convolutional filters to use
    nb_filters = 8
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 4
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(ELU())
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(ELU())

    sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


model = create_model_v1(100, 100, 1)

print(model.layers[0])