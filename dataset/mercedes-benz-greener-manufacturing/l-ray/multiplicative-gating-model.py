import numpy as np
import math
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
import theano
import theano.tensor as T
from lasagne.layers import (InputLayer, ConcatLayer, ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer, batch_norm)
import lasagne
from lasagne.init import HeNormal

LEARNING_RATE = 5e-6
num_epochs = 801
n_train_split = 3000

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

class MultiplicativeGatingLayer(lasagne.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]

def highway_dense(incoming, Wh=lasagne.init.Orthogonal(), bh=lasagne.init.Constant(0.0),
                  Wt=lasagne.init.Orthogonal(), bt=lasagne.init.Constant(-4.0),
                  nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    # regular layer
    l_h = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
                               nonlinearity=nonlinearity)
    # gate layer
    l_t = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                               nonlinearity=T.nnet.sigmoid)
    
    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)

def build_highway(input_var=None, output_dim=10, num_hidden_units=10, num_hidden_layers=2):
    l_in = lasagne.layers.InputLayer(shape=(None, 511),input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.5)
#    l_hidden1 = lasagne.layers.DenseLayer(l_in, num_units=num_hidden_units)
#    l_hidden1_drop = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    l_current = l_in_drop
    for k in range(num_hidden_layers - 1):
        l_current = highway_dense(l_current)
#    l_current_drop = lasagne.layers.DropoutLayer(l_current, p=0.5)
    l_out = lasagne.layers.DenseLayer(l_current, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
    
    return l_out

class ParaDenseLayer(lasagne.layers.base.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(ParaDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "leaving no trailing axes for the dot product." %
                    (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "requesting more trailing axes than there are input "
                    "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                    "A DenseLayer requires a fixed input shape (except for "
                    "the leading axes). Got %r for num_leading_axes=%d." %
                    (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            input = input.flatten(num_leading_axes + 1)

#        activation = T.dot(input, self.W)
        activation = T.dot(input, 2.0 / (1.0 + np.exp(-3.0*self.W+5.0)) + 2.0 / (1.0 + np.exp(-3.0*self.W-5.0)) - 2.0 )
#        activation = T.dot(input, 4.04 / (1.0 + np.exp(-self.W)) - 2.02)
#        activation = T.dot(input, np.arctan(self.W))

        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)

def build_repara(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 511), input_var=input_var)

    l_hid1 = ParaDenseLayer(l_in, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
#    l_hid2 = ParaDenseLayer(l_hid1, num_units=5, nonlinearity=lasagne.nonlinearities.rectify)
#    l_hid3 = ParaDenseLayer(l_hid2, num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
#    l_hid4 = ParaDenseLayer(l_hid3, num_units=100, nonlinearity=lasagne.nonlinearities.rectify)

#    l_hid1 = lasagne.layers.DenseLayer(l_in,   num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
#    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

#    l_out = ParaDenseLayer(l_hid1, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
#    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return l_hid1



def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 511), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.25)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=50, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.25)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=20, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.25)
    
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=1, nonlinearity=lasagne.nonlinearities.rectify)

    return l_out

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
    y_value = float(l[:-1].split(',')[1])
    if y_value<170:
        X = []
    
        y_train.append([y_value])
        for i in range(2,10):
            c = [-0.5]*length[i]
            c[char_dict[i][l.split(',')[i]]] = 0.5
            X.extend(c)
        for i in col:
            X.append( float(l[:-1].split(',')[i])-0.5 )
        X_train.append(X)

del lines

X_train = np.array(X_train,dtype='float32')
y_train = np.array(y_train,dtype='float32')
X_train_0 = X_train[:n_train_split]
y_train_0 = y_train[:n_train_split]
X_test = X_train[n_train_split:]
y_test = y_train[n_train_split:]
#X_train_0, X_test, y_train_0, y_test = train_test_split(X_train, y_train, test_size=0.2)
del X_train
del y_train
print('data ready with shapes:')
print(X_train_0.shape)
print(y_train_0.shape)
print(X_test.shape)
print(y_test.shape)

input_var = T.matrix('inputs') 
target_var = T.matrix('targets')

network = build_repara(input_var)
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
#test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
#test_loss = test_loss.mean()

#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
#val_fn = theano.function([input_var, target_var], test_loss)

n_model = 1

for i in range(n_model):
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train_0, y_train_0, 10, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        p_train = predict_fn(X_train_0)
#        train_r2 = r2_score(y_train_0, p_train)
#        if epoch%20==0:
#            print(epoch, '\t',train_err/train_batches, '\t',train_r2)
        p_test1 = predict_fn(X_test[:604])
        p_test2 = predict_fn(X_test[604:])
        p_test3 = predict_fn(X_test[300:904])
        train_r2 = r2_score(y_train_0, p_train)
        test_r2_1 = r2_score(y_test[:604], p_test1)
        test_r2_2 = r2_score(y_test[604:], p_test2)
        test_r2_3 = r2_score(y_test[300:904], p_test3)
        if epoch%20==0:
            print(epoch, '\t',train_err/train_batches, '\t',train_r2, '\t', test_r2_1, '\t', test_r2_2, '\t', test_r2_3)
print('final: ', train_err/train_batches, '\t', train_r2, '\t', test_r2_1, '\t', test_r2_2, '\t', test_r2_3)
del X_train_0
del y_train_0
del X_test
del y_test
del p_train
del p_test1
del p_test2

f = open('../input/test.csv', 'r')
lines = f.readlines()
del lines[0]
f.close()

X_test = []
test_id = []
for l in lines:
    X = []
    test_id.append(l.split(',')[0])
    for i in range(2,10):
        c = [-0.5]*length[i]
        key = l.split(',')[i-1]
        if key in char_set[i]:
            c[char_dict[i][key]] = 0.5
        X.extend(c)
    for i in col:
        X.append( float(l[:-1].split(',')[i-1])-0.5 )
    X_test.append(X)

X_test = np.array(X_test,dtype='float32')

p_test = predict_fn(X_test)

outfile = open('sub.csv', 'w')
outfile.write('ID,y\n')
n_test = len(test_id)
for i in range(n_test):
    outfile.write(test_id[i] + ',' + str(p_test[i][0])+'\n' )

outfile.close()

#np.savez('multiplicative.npz', *lasagne.layers.get_all_param_values(network))
