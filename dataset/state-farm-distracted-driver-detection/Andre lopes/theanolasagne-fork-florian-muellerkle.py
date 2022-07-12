import glob
import math
import os
import time

import cv2
import lasagne
import numpy as np
import pandas as pd
import theano
from lasagne.updates import nesterov_momentum
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from theano import tensor as T

'''
Loading data functions
'''

image_width = 24 *3
image_height = 32 * 3
imageSize = image_width * image_height

color_channels = 1
num_features = imageSize * color_channels


def load_train_cv(encoder):
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:

            if color_channels == 1:
                img = cv2.imread(fl, 0)
            elif color_channels == 3:
                img = cv2.imread(fl)

            img = cv2.resize(img, (image_width, image_height))
            # img = img.transpose(2, 0, 1)
            img = np.reshape(img, (1, num_features))

            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = encoder.fit_transform(y_train).astype('int32')

    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(X_train.shape[0], color_channels, image_height, image_width).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], color_channels, image_height, image_width).astype('float32') / 255.

    return X_train, y_train, X_test, y_test, encoder


def load_test():
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)

    for fl in files:
        flbase = os.path.basename(fl)

        if color_channels == 1:
            img = cv2.imread(fl, 0)
        elif color_channels == 3:
            img = cv2.imread(fl)

        img = cv2.resize(img, (image_height, image_width))
        # img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1

        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    X_test = X_test.reshape((X_test.shape[0], color_channels, image_height, image_width)).astype('float32') / 255.

    return X_test, X_test_id


'''
Lasagne Model ZFTurboNet and Batch Iterator
'''


def ZFTurboNet(input_var=None):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, color_channels, image_height, image_width),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=8, filter_size=(2, 2), pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=8, filter_size=(2, 2), pad=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


"""
Set up all theano functions
"""
BATCHSIZE = 32
LR = 0.1
ITERS = 5

np.random.seed(2016)
#lasagne.random.set_rng(np.random.seed(2016))

X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ZFTurboNet(X)
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize when using cat cross entropy our Y should be ints not one-hot
loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

# set up loss functions for validation dataset
valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
valid_loss = valid_loss.mean()

valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

# get parameters from network and set up sgd with nesterov momentum to update parameters, l_r is shared var so it can be changed
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = nesterov_momentum(loss, params, learning_rate=LR, momentum=0.9)

# set up training and prediction functions
train_fn = theano.function(inputs=[X, Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X, Y], outputs=[valid_loss, valid_acc])

# set up prediction function
predict_proba = theano.function(inputs=[X], outputs=output_test)

'''
load training data and start training
'''
encoder = LabelEncoder()

# load the training and validation data sets
train_X, train_y, test_X, test_y, encoder = load_train_cv(encoder)
print('Train shape:', train_X.shape, 'Test shape:', test_X.shape)

import cv2
aux_img = train_X[0]
aux_img = aux_img.reshape(image_height,image_width)
aux_img = aux_img * np.float32(255)
cv2.imwrite("test.jpg",aux_img)

aux_img = train_X[10]
aux_img = aux_img.reshape(image_height,image_width)
aux_img = aux_img * np.float32(255)
cv2.imwrite("test1.jpg",aux_img)

# loop over training functions for however many iterations, print information while training
try:
    for epoch in range(ITERS):
        # do the training
        start = time.time()
        # training batches
        train_loss = []
        for batch in iterate_minibatches(train_X, train_y, BATCHSIZE):
            inputs, targets = batch
            train_loss.append(train_fn(inputs, targets))
        train_loss = np.mean(train_loss)
        # validation batches
        valid_loss = []
        valid_acc = []
        for batch in iterate_minibatches(test_X, test_y, BATCHSIZE):
            inputs, targets = batch
            valid_eval = valid_fn(inputs, targets)
            valid_loss.append(valid_eval[0])
            valid_acc.append(valid_eval[1])
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)
        # get ratio of TL to VL
        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print('iter:', epoch, '| TL:', np.round(train_loss, decimals=3), '| VL:', np.round(valid_loss, decimals=3), '| Vacc:', np.round(valid_acc, decimals=3), '| Ratio:',
              np.round(ratio, decimals=2), '| Time:', np.round(end, decimals=1))

except KeyboardInterrupt:
    pass

'''
Make Submission
'''
# load data
X_test, X_test_id = load_test()

# make predictions
PRED_BATCH = 1
predictions = []
for j in range((X_test.shape[0] + PRED_BATCH - 1) // PRED_BATCH):
    sl = slice(j * PRED_BATCH, (j + 1) * PRED_BATCH)
    X_batch = X_test[sl]
    predictions.extend(predict_proba(X_batch))

predictions = np.array(predictions)


def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('submission_ZFTurboNet.csv', index=False)


create_submission(predictions, X_test_id)
