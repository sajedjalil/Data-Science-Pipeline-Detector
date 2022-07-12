# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import pickle
from keras.layers.noise import GaussianDropout
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ThresholdedReLU, LeakyReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.metrics import r2_score
import keras.backend as K
from keras import optimizers
from keras import  regularizers
# from utils import  *
from keras.constraints import maxnorm
import matplotlib.pyplot as plt


def extract_features(model, X, layer):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[layer].output])
    layer_output = get_layer_output([X, 1])[0]

    return np.asarray(layer_output)


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
Y_train = df_train.y
df_train = df_train.drop(['y', 'ID'], axis = 1)
id_test = df_test.ID
df_test = df_test.drop(['ID'], axis = 1)



num_train = len(df_train)
X_all = pd.concat([df_train, df_test])


for column in X_all.columns:
    cardinality = len(np.unique(X_all[column]))
    if cardinality == 1:
        X_all.drop(column, axis=1)  # Column with only one value is useless so we drop it
    if cardinality > 2:  # Column is categorical
        mapper = lambda x: sum([ord(digit) for digit in x])
        X_all[column] = X_all[column].apply(mapper)

X_train = X_all[:num_train].values
X_test = X_all[num_train:].values

print(X_train.shape)
print(X_test.shape)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

act_fun = 'relu'

def nn_model():
    model = Sequential()

    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))

    model.add(Dense(int(1.3 * X_train.shape[1]), kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))



    model.add(Dense(X_train.shape[1] // 16, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))
    model.add(Dropout(0.05))

    model.add(Dense(X_train.shape[1] // 32, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))

    model.add(Dense(X_train.shape[1] // 64, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))

    model.add(Dense(X_train.shape[1] // 32,  kernel_constraint=maxnorm(3)))
    model.add(Activation(act_fun))


    model.add(Dense(X_train.shape[1] // 16, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))
    model.add(Dropout(0.05))


    model.add(Dense(int(1.3 * X_train.shape[1]), kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))


    model.add(Dense(X_train.shape[1], activation='relu')) #activity_regularizer=regularizers.l1(0.01))
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[r2_keras, root_mean_squared_error])
    return (model)

    return model


model = nn_model()
print(model.summary())

"""
count = 0
for layer in model.layers:
    print count, layer
    count += 1
"""
history = model.fit(X_train, X_train, nb_epoch = 200, batch_size=200, verbose = 1, validation_split = 0.2, shuffle=True)


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r2')
plt.ylabel('r2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

X_tr = extract_features(model, X_train, 13)
X_te = extract_features(model, X_test, 13)
pickle.dump(X_tr, open("train_ae.p", "wb"))
pickle.dump(X_te, open("test_ae.p", "wb"))
