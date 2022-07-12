###########################################################
# See https://www.kaggle.com/c/whats-cooking/forums/t/16657/deep-cooking/93391
# for further information.
###########################################################


# This script contains some elements from Kappa's script and valerio orfano's script
# https://www.kaggle.com/c/whats-cooking/forums/t/16538/simple-theano-script-with-0-79033-on-leaderboard

import gc
import os
import pickle
import json

import numpy as np
import pandas as pd

from collections import OrderedDict

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from itertools import chain

import theano
import theano.tensor as T

import lasagne as nn


###########################################################
# auxiliary functions for nn

def sgd(loss, all_parameters, learning_rate):
    all_grads = [theano.grad(loss, param) for param in all_parameters]
    updates = []

    for param_i, grad_i in zip(all_parameters, all_grads):
        v = - learning_rate * grad_i

        # clip from 0.0 to 1.0
        updates.append((param_i, T.clip(param_i + v, 0.0, 1.0)))
    return updates


def get_param_values(params):
    return [p.get_value() for p in params]

def set_param_values(params, param_values):
    for p, pv in zip(params, param_values):
        p.set_value(pv)

def normalize_input(X):
    return (X.T / np.sum(X, axis=1)).T



###########################################################
# load and preprocess data

print('loading data ...')

input_dir = '../input/'

# train
with open(os.path.join(input_dir, 'train.json')) as train_f:
    train_data = json.loads(train_f.read())

X_train = [x['ingredients'] for x in train_data]
X_train = [dict(zip(x,np.ones(len(x)))) for x in X_train]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train).toarray()
X_train = normalize_input(X_train)
X_train = X_train.astype(np.float32)

feature_names = np.array(vec.feature_names_)

lbl = LabelEncoder()

y_train = [y['cuisine'] for y in train_data]
y_train = lbl.fit_transform(y_train).astype(np.int32)

label_names = lbl.classes_ 
for i, l in enumerate(label_names):
    print('i: {}, l: {}'.format(i, l))

## sample here for memory restrictions
#idx = np.random.choice(X_train.shape[0], 4096)
#X_train = X_train[idx]
#y_train = y_train[idx]

# for memory restrictions use only british or chinese
idx = np.logical_or(y_train == 1, y_train ==3 )
y_train = y_train[idx]
X_train = X_train[idx]

print('num_saples: {}'.format( y_train.shape[0] ))


###########################################################
# train nn for weights, this code is too bad

def get_nn_params(X_train, y_train):
    LEARNING_RATE = 0.01
    OUTPUT_DIM = label_names.size # = 20
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    
    # pad samples
    n_train = X_train.shape[0]
    
    # theano based neural network
    # (1) network
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')
    
    activation = nn.nonlinearities.rectify
    
    l_in = nn.layers.InputLayer(input_var=X_batch, shape=(BATCH_SIZE, X_train.shape[1]),)
    l_hidden0_dropout = nn.layers.DropoutLayer(l_in, p=0.0)
    
    l_hidden1 = nn.layers.DenseLayer( l_hidden0_dropout, num_units=256,
        nonlinearity=activation, W=nn.init.GlorotUniform(),)
    l_hidden1_dropout = nn.layers.DropoutLayer(l_hidden1, p=0.5)
    
    #l_hidden2 = nn.layers.DenseLayer( l_hidden1_dropout, num_units=128,
    #    nonlinearity=activation, W=nn.init.GlorotUniform(),)
    #l_hidden2_dropout = nn.layers.DropoutLayer(l_hidden2, p=0.5)
    
    # classifier
    l_out = nn.layers.DenseLayer( l_hidden1_dropout, num_units=OUTPUT_DIM,
        nonlinearity=nn.nonlinearities.softmax, W=nn.init.GlorotUniform(),) 
    
    
    # (2) i/o 
    X_shared = theano.shared(np.zeros((1, 1,), dtype=theano.config.floatX))
    y_shared = theano.shared(np.zeros((1,), dtype=theano.config.floatX))
    y_shared_casted = T.cast(y_shared, 'int32')
    
    batch_index = T.lscalar('batch_index')
    
    
    # (3) loss, outputs, updates
    learning_rate = theano.shared(np.array(LEARNING_RATE, dtype=theano.config.floatX))
    
    all_params = nn.layers.get_all_params(l_out)
    
    loss_train = T.mean(-T.log( nn.layers.get_output(l_out) )[T.arange(y_batch.shape[0]), y_batch])
    loss_eval = T.mean(-T.log( nn.layers.get_output(l_out, deterministic=True) )[T.arange(y_batch.shape[0]), y_batch])
    
    pred = T.argmax( nn.layers.get_output(l_out, deterministic=True), axis=1)
    pred_proba = nn.layers.get_output(l_out, deterministic=True)
    
    updates = nn.updates.nesterov_momentum( loss_train, all_params, learning_rate,)
    
    givens = {
        X_batch: X_shared[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE],
        y_batch: y_shared_casted[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE],
        }
        
    train = theano.function( [batch_index], [loss_train], updates=updates, givens=givens)
    test = theano.function( [batch_index], [loss_eval, pred_proba], givens=givens)
    
    # train
    print('start training')
    for e in range(NUM_EPOCHS):
    
        # shuffle and pad train sample
        idx = np.arange(y_train.size)
        np.random.shuffle(idx)
        idx = idx[:(idx.shape[0] / BATCH_SIZE * BATCH_SIZE)]
    
        X_shared.set_value(X_train[idx].astype(np.float32))
        y_shared.set_value(y_train[idx].astype(np.float32))
    
    
        train_losses = []
    
        for b in range(int(idx.shape[0] / BATCH_SIZE)):
            (train_loss,) = train(b)
            train_losses.append(train_loss)
            (_, p) = test(b)
    
        mean_train_loss = np.mean(train_losses)
        print('  epoch: {}, loss: {}'.format(e, mean_train_loss))
    
    return get_param_values(all_params)

print('get nn params ...')
nn_params = get_nn_params(X_train, y_train)


###########################################################
# NN for output

LEARNING_RATE = 0.01
OUTPUT_DIM = label_names.size # = 20

BATCH_SIZE = 1 #256

# theano based neural network

# (2) i/o 
X_shared = theano.shared(np.zeros((1, 1,), dtype=theano.config.floatX))
y_shared = theano.shared(np.zeros((1, 1,), dtype=theano.config.floatX))
#y_shared = theano.shared(np.zeros((1,), dtype=theano.config.floatX))
#y_shared_casted = T.cast(y_shared, 'int32')

batch_index = T.lscalar('batch_index')

activation = nn.nonlinearities.rectify

l_in = nn.layers.InputLayer(input_var=X_shared, shape=(BATCH_SIZE, X_train.shape[1]),)
l_hidden0_dropout = nn.layers.DropoutLayer(l_in, p=0.0)

l_hidden1 = nn.layers.DenseLayer( l_hidden0_dropout, num_units=256,
    nonlinearity=activation, W=nn.init.GlorotUniform(),)
l_hidden1_dropout = nn.layers.DropoutLayer(l_hidden1, p=0.5)

#l_hidden2 = nn.layers.DenseLayer( l_hidden1_dropout, num_units=128,
#    nonlinearity=activation, W=nn.init.GlorotUniform(),)
#l_hidden2_dropout = nn.layers.DropoutLayer(l_hidden2, p=0.5)

# classifier
l_out = nn.layers.DenseLayer( l_hidden1_dropout, num_units=OUTPUT_DIM,
    nonlinearity=nn.nonlinearities.softmax, W=nn.init.GlorotUniform(),) 


# (3) loss, outputs, updates
learning_rate = theano.shared(np.array(LEARNING_RATE, dtype=theano.config.floatX))

all_params = nn.layers.get_all_params(l_out)

# load weights
#nn_params = pickle.load(open('nn_params.pkl'))
set_param_values(all_params, nn_params)

#loss_train = T.mean(-T.log( nn.layers.get_output(l_out) )[T.arange(y_shared_casted.shape[0]), y_shared_casted])
#loss_eval = T.mean(-T.log( nn.layers.get_output(l_out, deterministic=True) )[T.arange(y_shared_casted.shape[0]), y_shared_casted])


loss_train = T.mean( ( nn.layers.get_output(l_out) - y_shared) ** 2.0 )

#loss_train = T.mean( ( nn.layers.get_output(l_out) - y_shared) ** 2.0 ) \
#    + 0.0003 * T.mean(T.sum(abs(X_shared), axis=1)) 


#updates = nn.updates.nesterov_momentum( loss_train, all_params, learning_rate,)
#updates = nn.updates.sgd( loss_train, [X_shared], learning_rate,)
updates = sgd( loss_train, [X_shared], learning_rate,) # sgd with clip

train = theano.function([], [loss_train], updates=updates,)



###########################################################
# html for output

o = '''
<!DOCTYPE html>
<html>

<head>
<meta charset="UTF-8"> 

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap-theme.min.css">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>

</head>

<body>

<!---
<h3>Title</h3>

<p>Description</p>
--->

'''

table_head = '''
<h4>{}</h4>

<ul>
'''

table_row = '''
<li>{} <small>{}</small></li>
'''

table_foot = '''
</ul>
'''


def write(o, X, title=''):
    o += table_head.format(title)
    
    X = np.mean(X, axis=0)

    idx = np.argsort(-X)[:10]

    for i in idx:
        o += table_row.format(feature_names[i], X[i])
    
    o += table_foot
    
    return o



###########################################################
# main


## (1.1) "mean" of british
#X_chinese = X_train[y_train == 1]
#o = write(o, X_chinese, '"Mean" of british cuisine')


# (1.2) nn 
X_noise = np.random.uniform(low=0.0, high=1.0,
    size=(BATCH_SIZE, X_train.shape[1])) * 0.001
y_noise = np.zeros([BATCH_SIZE, OUTPUT_DIM])

y_noise[:,1] = 1.0

X_shared.set_value(X_noise.astype(np.float32))
y_shared.set_value(y_noise.astype(np.float32))

print('start training')
for i in range(30):
    (l,) = train()
    print('epoch: {}, loss: {}'.format(i, l))

X_result = X_shared.get_value()
o = write(o, X_result, 'British cuisine by NN model')


## (2.1) "mean" of chinese
#X_chinese = X_train[y_train == 3]
#o = write(o, X_chinese, '"Mean" of chinese cuisine')


# (2.2) nn 
X_noise = np.random.uniform(low=0.0, high=1.0,
    size=(BATCH_SIZE, X_train.shape[1])) * 0.001
y_noise = np.zeros([BATCH_SIZE, OUTPUT_DIM])

y_noise[:,3] = 1.0

X_shared.set_value(X_noise.astype(np.float32))
y_shared.set_value(y_noise.astype(np.float32))

print('start training')
for i in range(30):
    (l,) = train()
    print('epoch: {}, loss: {}'.format(i, l))

X_result = X_shared.get_value()
o = write(o, X_result, 'Chinese cuisine by NN model')




###########################################################
# output


o += '''
</html>
'''

with open("results.html","wb") as outfile:
    outfile.write(o.encode("utf-8"))
