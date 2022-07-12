#! /usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random


train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

# Really simple data preparation
y_train = pd.get_dummies(train[["type"]], prefix="")
train.drop("type", inplace=True, axis=1)

train_test = pd.concat([train, test], axis=0)

# It looks like the color actually is just noise, and does not give any signal to the monster-class.
# Comment one of these lines.
#train_test = pd.get_dummies( train_test, columns=["color"], drop_first=False)
train_test.drop("color", inplace=True, axis=1)

X_train = train_test.iloc[:len(y_train)]
X_test  = train_test.iloc[len(y_train):]

# Clean up
del train_test
del train
del test

## A dead simple neural network class in Python+Numpy. Plain SGD, and no regularization.
def sigmoid(X):
    return 1.0 / ( 1.0 + np.exp(-X) )

def softmax(X):
    _sum = np.exp(X).sum()
    return np.exp(X) / _sum

class neuralnet(object):
    def __init__(self, num_input, num_hidden, num_output):
        self._W1 = (np.random.random_sample((num_input, num_hidden)) - 0.5).astype(np.float32)
        self._b1 = np.zeros((1, num_hidden)).astype(np.float32)
        self._W2 = (np.random.random_sample((num_hidden, num_output)) - 0.5).astype(np.float32)
        self._b2 = np.zeros((1, num_output)).astype(np.float32)

    def forward(self,X):
        net1 = np.matmul( X, self._W1 ) + self._b1
        y = sigmoid(net1)
        net2 = np.matmul( y, self._W2 ) + self._b2
        z = softmax(net2)
        return z,y

    def backpropagation(self, X, target, eta):
        z, y = self.forward(X)
        d2 = (z - target)
        d1 = y*(1.0-y) * np.matmul(d2, self._W2.T)
        # The updates are done within this method. This more or less implies
        # utpdates with Stochastic Gradient Decent. Let's fix that later.
        # TODO: Support for full batch and mini-batches etc.
        self._W2 -= eta * np.matmul(y.T,d2)
        self._W1 -= eta * np.matmul(X.reshape((-1,1)),d1)
        self._b2 -= eta * d2
        self._b1 -= eta * d1
        
# Some hyper-parameters to tune.
num_hidden = 12
n_epochs   = 2000
eta        = 0.01
# Create the net.
nn = neuralnet( X_train.shape[1], num_hidden, y_train.shape[1])


# (EDIT: It's much faster to convert the dataframes to numpy arrays and then iterate)
X = np.array(X_train, dtype=np.float32)
Y = np.array(y_train, dtype=np.float32)
for epoch in range(n_epochs):
    for monster, target in zip(X,Y):
        nn.backpropagation( monster, target, eta)
        
with open('submission-{}-hidden.csv'.format(num_hidden), 'w') as f:
    f.write("id,type\n")
    for index, monster in X_test.iterrows():
        probs = nn.forward( np.array(monster, dtype=np.float32))[0]
        f.write("{},{}\n".format(index, y_train.columns.values[np.argmax(probs)][1:]))
