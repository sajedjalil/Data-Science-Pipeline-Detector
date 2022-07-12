
# coding: utf-8

# CDiscount Image Classification Challenge

import io
import time
import threading
import itertools


import pandas as pd
import bson
import numpy as np
from scipy.misc import imresize, imread
import matplotlib.pyplot as plt
#import lycon

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam

import tensorflow as tf


# Get the category ID for each document in the training set.
documents = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
categories = [(d['_id'], d['category_id']) for d in documents]
categories = pd.DataFrame(categories, columns=['id', 'cat'])

# Create a label encoder for all the labels found
labelencoder = LabelEncoder()
labelencoder.fit(categories.cat.unique().ravel())


def grouper(n, iterable):
    '''
    Given an iterable, it'll return size n chunks per iteration.
    Handles the last chunk too.
    '''
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def get_features_label(documents, batch_size=32, return_labels=True):
    '''
    Given a document return X, y
    
    X is scaled to [0, 1] and consists of all images contained in document.
    y is given an integer encoding.
    '''
    
    
    for batch in grouper(batch_size, documents): 
        images = []
        labels = []

        for document in batch:
            category = document.get('category_id', '')
            img = document.get('imgs')[0]
            data = io.BytesIO(img.get('picture', None))
            im = imread(data)

            if category:    
                label = labelencoder.transform([category])
            else:
                label = None

            im = im.astype('float32') / 255.0

            images.append(im)
            labels.append(label)

        if return_labels:
            yield np.array(images), np.array(labels)
        else:
            yield np.array(images)


def create_model(num_classes=None, input_shape=(180, 180, 3)):
    '''
    Create the NN model, compile it and return the model object.
    
    Initial model shamelessly stolen from Keras docs.
    '''    
    import tensorflow as tf 
    
    model = Sequential()

    # Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.75))

    # Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.75))
    
    # Layer 3
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    
    # So we can look at the progress on Tensorboard
    tb_callback = keras.callbacks.TensorBoard(
        log_dir='./logs/{}-{}'.format(time.time(), ' '.join(l.name for l in model.layers))
    )
    
    return model, [tb_callback]

# Create generator expressions for training and testing data
# Keras' fit_generator can use these to fit the data without 
# loading everything in memory.

generator = get_features_label(bson.decode_file_iter(open('../input/train_example.bson', 'rb')), batch_size=2)

# Now we can train this model.
model, callbacks = create_model(num_classes=len(labelencoder.classes_))

model.fit_generator(generator=generator,
                    epochs=1,
                    steps_per_epoch=2)