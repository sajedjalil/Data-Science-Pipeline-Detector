# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math as mt
# preprocessing/decomposition
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
# It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error

# feature selection
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold

# define path to save model
import os

model_path = 'keras_model.h5'

# to make results reproducible
seed = 123

# Read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Remove the outlier
# train=train[train.y<250]

# save IDs for submission
id_test = test['ID'].copy()

y = train.y.values
###########################
# DATA PREPARATION
###########################

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

#total = total.filter(items=['ID', 'X0'])
# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False  # you can set it = True to ommit multicollinearity (crucial for linear models)
)

print('oh-encoded shape: {}'.format(dummies.shape))

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1),  # drop old
        dummies  # append them one-hot-encoded
    ],
    axis=1  # column-wise
)

print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train].drop(['ID','y'], axis=1), total[~is_train].drop(['ID','y'], axis=1)

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


#########################################################################################################################################
# GENERATE MODEL
# The Keras wrappers require a function as an argument.
# This function that we must define is responsible for creating the neural network model to be evaluated.
# Below we define the function to create the baseline model to be evaluated.
# The network uses good practices such as the rectifier activation function for the hidden layer.
# No activation function is used for the output layer because it is a regression problem and we are interested in predicting numerical
# values directly without transform.# The efficient ADAM optimization algorithm is used and a mean squared error loss function is optimized.
# This will be the same metric that we will use to evaluate the performance of the model.
# It is a desirable metric because by taking the square root gives us an error value we can directly understand in the context of the problem.
##########################################################################################################################################

def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# Base model architecture definition.
# Dropout is a technique where randomly selected neurons are ignored during training.
# They are dropped-out randomly. This means that their contribution to the activation.
# of downstream neurons is temporally removed on the forward pass and any weight updates are
# not applied to the neuron on the backward pass.
# More info on Dropout here http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# BatchNormalization, Normalize the activations of the previous layer at each batch, i.e. applies a transformation
# that maintains the mean activation close to 0 and the activation standard deviation close to 1.
def model():
    model = Sequential()
    # Input layer with dimension input_dims and hidden layer i with input_dims neurons.
    model.add(Dense(input_dims, input_dim=input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation("linear"))
    # Hidden layer
    model.add(Dense(input_dims//2, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Activation("linear"))
    # Output Layer.
    model.add(Dense(1))

    # Use a large learning rate with decay and a large momentum.
    # Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile this model
    rms = RMSprop(lr=0.0025, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error',  # one may use 'mean_absolute_error' as alternative
                  optimizer=adam,
                  metrics=[r2_keras, "mse"]  # you can add several if needed
                  )

    # Visualize NN architecture
    print(model.summary())
    return model


# initialize input dimension

input_dims = train.shape[1]
# input_dims = train_reduced.shape[1]

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model,
    nb_epoch=50,
    batch_size=80,
    verbose=1
)

# X, y preparation
X = train.values
X_test = test.values
print('\nTrain shape No Feature Selection: {}\nTest shape No Feature Selection: {}'.format(X.shape, X_test.shape))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def sum_of_square_deviation(numbers, mean):
    return float(1 / len(numbers) * sum((x - mean) ** 2 for x in numbers))

# fit estimator
history = estimator.fit(
    X,
    y,
    epochs=50,
    verbose=2,
    shuffle=True
)

# list all data in history
print(history.history.keys())

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y, estimator.predict(X)) ** 0.5))  # mse train
print('R^2 train: {}'.format(r2_score(y, estimator.predict(X))))  # R^2 train

# predict results
res = estimator.predict(X_test).ravel()
print(res)

# create df and convert it to csv
output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv('keras-baseline.csv', index=False)