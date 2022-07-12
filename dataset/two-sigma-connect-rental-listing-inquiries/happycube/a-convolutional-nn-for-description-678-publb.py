'''

This isn't well organized/edited yet, but it gets ~.68 on the public LB using only Description with 
no English-specific handling, just a bespoke convnet.

v2:  a bit better organized, still quick+dirty though :)  functionally the same as v1.

'''

# a lot of this isn't needed, but this isn't Go so they can stay.
import numpy as np
import pandas as pd

import xgboost as xgb

import scipy.stats
from scipy import sparse

import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing

import tempfile
from multiprocessing import Process, Queue
import pickle
import os

from operator import itemgetter
from collections import defaultdict, Counter

import tensorflow as tf

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Conv2D, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D
from keras.layers import Reshape
import keras

from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Input

from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Input

from keras.layers.recurrent import GRU
from keras.layers import Flatten

# from https://github.com/martin-gorner/tensorflow-rnn-shakespeare
# (an excellent sample RNN and talk, take a look at that if you're into tensorflow!)

# size of the alphabet that we work with
ALPHASIZE = 98

# Specification of the supported alphabet (subset of ASCII-7)
# 10 line feed LF
# 32-64 numbers and punctuation
# 65-90 upper-case letters
# 91-97 more punctuation
# 97-122 lower-case letters
# 123-126 more punctuation
def convert_from_alphabet(a):
    """Encode a character
    :param a: one character
    :return: the encoded value
    """
    if a == 9:
        return 1
    if a == 10:
        return 127 - 30  # LF
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0  # unknown
    
def encode_text(s):
    """Encode a string.
    :param s: a text string
    :return: encoded list of code points
    """
    return list(map(lambda a: convert_from_alphabet(ord(a)), s))

# base loading and CV code
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

traintest = pd.concat([train, test])

MAXDESC = 1000 #around the mean (or mean+std, i forget)

traintest['desc_enc'] = [np.array(encode_text(d[1].description[:MAXDESC])) for d in traintest.iterrows()]

traintest['interest_level'] = traintest['interest_level'].replace({'low': 0, 'medium': 1, 'high': 2})
traintest['interest_low'] = traintest['interest_level'] == 0
traintest['interest_med'] = traintest['interest_level'] == 1
traintest['interest_high'] = traintest['interest_level'] == 2

def oneheat(cold):
    output = np.zeros((len(cold), MAXDESC, int(np.max(cold)+ 1)), dtype=np.int8)
    print(output.shape)

    for i, c in enumerate(cold):
        for i2, c2 in enumerate(c):
#            print(i2, int(c2))
            output[i, i2, int(c2)] = 1

    return output

# build up a 1000-length array.  probably could do this in oneheat, don't care.
desc_array = np.zeros((traintest.shape[0], MAXDESC))

for i, s in enumerate(traintest.desc_enc.values):
        desc_array[i][:len(s)] = s

hot = oneheat(desc_array)
# Pandas doesn't like direct assignment for whatever reason...
traintest['desc_hot'] = [t for i, t in enumerate(hot)]

del hot # should save some memory somewhere, right?

train = traintest.loc[~traintest.interest_level.isnull()]
test = traintest.loc[traintest.interest_level.isnull()]

def split_train(train, folds=4, seed=0):
    skf = sklearn.model_selection.StratifiedKFold(n_splits = folds, shuffle = True, random_state=seed)
    skf_gen = skf.split(train, train.interest_level)

    skf_indexes = [c for c in skf_gen]

    cv_train = [train.iloc[c[0]] for c in skf_indexes]
    cv_valid = [train.iloc[c[1]] for c in skf_indexes]
    
    return cv_train, cv_valid

cv_train, cv_valid = split_train(train.copy())

# build the 1D subset of the keras (functional) model
def buildconv1d(inp, num, size, basename):
    conv1 = Conv1D(num, size, padding='valid', activation='relu', strides=1, name=basename+"_conv1")(inp)
    drop1 = Dropout(0.1, name=basename+'_drop1')(conv1)

    pool1 = GlobalMaxPooling1D(name=basename+'_pool1')(drop1)

    splat = Flatten()(drop1)

    dense1 = Dense(160, activation='relu', name=basename+'_dense')(pool1)
    dense1drop = Dropout(0.2, name=basename+'_densedrop')(dense1)

    return dense1drop

def run_nn(usetrain, usetest):
    stringsize = 1000

    y_train = usetrain[['interest_low', 'interest_med', 'interest_high']].values
    y_test = usetest[['interest_low', 'interest_med', 'interest_high']].values
    
    inshape = 97
    
    # the one hot values need to be a single array in the proper arrangement.  this is RAM hungry,
    # but the computer I run it on has 48GB. So :P. ;)
    
    tmp = np.concatenate(usetrain.desc_hot.values)
    xhot_train = np.reshape(tmp, [tmp.shape[0] // stringsize, stringsize, inshape])

    tmp = np.concatenate(usetest.desc_hot.values)
    xhot_test = np.reshape(tmp, [tmp.shape[0] // stringsize, stringsize, inshape])

    inputs = Input(shape=(stringsize, inshape,))
    reshape1 = Reshape((-1, stringsize, ))(inputs)

    path = []
    path.append(buildconv1d(inputs, 80, 8, 'path1'))
    path.append(buildconv1d(inputs, 80, 16, 'path2'))
    path.append(buildconv1d(inputs, 80, 32, 'path3'))

    dense1 = Dense(256, activation='relu')(concatenate(path))
    drop1 = Dropout(0.2)(dense1)

    output = Dense(3, activation='softmax', name='output')(drop1)

    model = Model(inputs=inputs, outputs=output)
    #print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    for e in range(0, 3):
        model.fit(xhot_train, y_train, batch_size=256, epochs=1, verbose=2, validation_data=(xhot_test, y_test))
        
    tpreds = model.predict(xhot_test)
        
    return model, tpreds

rv = [run_nn(cv_train[i], cv_valid[i]) for i in range(0 , 4)]
rv_test = run_nn(train, test)

'''
Train on 37013 samples, validate on 12339 samples
Epoch 1/1
80s - loss: 0.7565 - categorical_accuracy: 0.6971 - val_loss: 0.7147 - val_categorical_accuracy: 0.7039
Train on 37013 samples, validate on 12339 samples
Epoch 1/1
60s - loss: 0.6807 - categorical_accuracy: 0.7103 - val_loss: 0.6876 - val_categorical_accuracy: 0.7064
Train on 37013 samples, validate on 12339 samples
Epoch 1/1
61s - loss: 0.6229 - categorical_accuracy: 0.7319 - val_loss: 0.6848 - val_categorical_accuracy: 0.7078
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
65s - loss: 0.7583 - categorical_accuracy: 0.6934 - val_loss: 0.7058 - val_categorical_accuracy: 0.7006
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
60s - loss: 0.6809 - categorical_accuracy: 0.7099 - val_loss: 0.6799 - val_categorical_accuracy: 0.7098
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
60s - loss: 0.6303 - categorical_accuracy: 0.7299 - val_loss: 0.6804 - val_categorical_accuracy: 0.7109
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
69s - loss: 0.7625 - categorical_accuracy: 0.6921 - val_loss: 0.7119 - val_categorical_accuracy: 0.6980
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
68s - loss: 0.6838 - categorical_accuracy: 0.7091 - val_loss: 0.6810 - val_categorical_accuracy: 0.7094
Train on 37014 samples, validate on 12338 samples
Epoch 1/1
68s - loss: 0.6233 - categorical_accuracy: 0.7308 - val_loss: 0.6874 - val_categorical_accuracy: 0.7149
Train on 37015 samples, validate on 12337 samples
Epoch 1/1
72s - loss: 0.7590 - categorical_accuracy: 0.6929 - val_loss: 0.7090 - val_categorical_accuracy: 0.7063
Train on 37015 samples, validate on 12337 samples
Epoch 1/1
68s - loss: 0.6826 - categorical_accuracy: 0.7108 - val_loss: 0.7203 - val_categorical_accuracy: 0.7018
Train on 37015 samples, validate on 12337 samples
Epoch 1/1
68s - loss: 0.6183 - categorical_accuracy: 0.7318 - val_loss: 0.6843 - val_categorical_accuracy: 0.7062
Train on 49352 samples, validate on 74659 samples
Epoch 1/1
146s - loss: 0.7435 - categorical_accuracy: 0.6988 - val_loss: 0.0000e+00 - val_categorical_accuracy: 0.9216
Train on 49352 samples, validate on 74659 samples
Epoch 1/1
145s - loss: 0.6735 - categorical_accuracy: 0.7115 - val_loss: 0.0000e+00 - val_categorical_accuracy: 0.8975
Train on 49352 samples, validate on 74659 samples
Epoch 1/1
149s - loss: 0.6185 - categorical_accuracy: 0.7332 - val_loss: 0.0000e+00 - val_categorical_accuracy: 0.7307
'''

# now save everything

basename = 'keras402'

import pickle

for i, r in enumerate(rv):
    r[0].save('{0}-train{1}.mdl'.format(basename, i))

rv_test[0].save(basename + 'test.mdl')

for i, r in enumerate(rv):
    pickle.dump(r[1], open('{0}-train{1}p.pkl'.format(basename, i), 'wb'))

pickle.dump(rv_test[1], open(basename + '-testp.pkl', 'wb'))

cv_str = []
for i, v in enumerate(cv_valid):
    cv_str.append(v[['listing_id', 'interest_level']].copy())
    cv_str[-1]['low'] = rv[i][1][:,0]
    cv_str[-1]['medium'] = rv[i][1][:,1]
    cv_str[-1]['high'] = rv[i][1][:,2]

df_cv = pd.concat(cv_str)

print(sklearn.metrics.log_loss(df_cv.interest_level, df_cv[['low', 'medium', 'high']].values))
# 0.684218378082


df_testout = test[['listing_id']].copy()
df_testout['low'] = rv_test[1][:,0]
df_testout['medium'] = rv_test[1][:,1]
df_testout['high'] = rv_test[1][:,2]

df_testout.to_csv('sub-{0}.csv.gz'.format(basename), index=False, compression='gzip')
# this run got .69437 LB - it's pretty random!
