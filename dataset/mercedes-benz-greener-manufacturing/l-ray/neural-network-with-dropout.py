import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
import theano
import theano.tensor as T
from lasagne.layers import (InputLayer, ConcatLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, batch_norm)
import lasagne
from lasagne.init import HeNormal

LEARNING_RATE = 5e-6

def Rvalue(pred, target):
    assert len(pred)==len(target)
    mu = target.mean(axis=0)
    r = 1.0 -((target-pred)**2).sum() / ((target-mu)**2).sum()
    return np.sign(r) * math.sqrt(abs(r))
    

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

f = open('../input/train.csv', 'r')
lines = f.readlines()
f.close()
head = lines[0]
del lines[0]
n_train = len(lines)
n_dim = len(lines[0].split(','))

char_set = []
char_dict = []

for _ in range(10):
    char_set.append(set())
    char_dict.append(dict())

for l in lines:
    for idx in range(2,10):
        char_set[idx].add(l[:-1].split(',')[idx])

length = [0]*10
tot_length = 0
for i in range(2,10):
    length[i] = len(char_set[i])
    tot_length += len(char_set[i])

#print(tot_length)
for i in range(2,10):
    v = 0
    for s in char_set[i]:
        char_dict[i][s] = v
        v += 1

mean = [0.0]*n_dim
for l in lines:
    for idx in range(10,n_dim):
        mean[idx] += float(l[:-1].split(',')[idx] )
ignorable = []
for i in range(10,n_dim):
    mean[i] /= n_train
    if abs(mean[i]-1.0)<1e-3 or abs(mean[i]-0.0)<1e-3:
        ignorable.append(i)

col = set(range(10,n_dim)) - set(ignorable)

np.random.shuffle(lines)
X_train = []
y_train = []
for l in lines:
    X = []
    y_train.append([float(l[:-1].split(',')[1])])
    for i in range(2,10):
        c = [-0.5]*length[i]
        c[char_dict[i][l.split(',')[i]]] = 0.5
        X.extend(c)
    for i in col:
        X.append( float(l[:-1].split(',')[i])-0.5 )
    X_train.append(X)

X_train = np.array(X_train,dtype='float32')
y_train = np.array(y_train,dtype='float32')
X_train_0, X_test, y_train_0, y_test = train_test_split(X_train, y_train, test_size=0.3)
del X_train
del y_train
print('data ready with shapes:')
print(X_train_0.shape)
print(y_train_0.shape)
print(X_test.shape)
print(y_test.shape)

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 511), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.25)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=100, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.25)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=40, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.25)
    
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)

    return l_out

input_var = T.matrix('inputs') 
target_var = T.matrix('targets')

network = build_mlp(input_var)
print('network output shape: ', network.output_shape)

prediction = lasagne.layers.get_output(network)

#l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4
#prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False, batch_norm_update_averages=False, batch_norm_use_averages=False)

#loss += l2_loss
loss = T.sum(lasagne.objectives.squared_error(prediction, target_var))
#loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=LEARNING_RATE, momentum=0.5)


test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], test_prediction)
test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
test_loss = test_loss.mean()

#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], test_loss)

n_model = 1
num_epochs = 500
for i in range(n_model):
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train_0, y_train_0, 20, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        p_train = predict_fn(X_train_0)
        p_test = predict_fn(X_test)
        if epoch%20==0:
            print(epoch, '\t',train_err/train_batches, '\t', r2_score(y_train_0, p_train), '\t', r2_score(y_test, p_test))
#print(r2_score(vote, y_test))