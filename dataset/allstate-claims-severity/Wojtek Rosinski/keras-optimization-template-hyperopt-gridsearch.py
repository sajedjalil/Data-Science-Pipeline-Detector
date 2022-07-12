# Works on python 2.7 on my computer.
# Template for optimization. 
# Parameters can be changed further, this is starting point for me.

### HYPEROPT ###

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys

import pandas as pd
import numpy as np

np.random.seed(6669)

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1l2, activity_l2

import tensorflow as tf
tf.python.control_flow_ops = tf

# Based on Faron's stacker. Thanks!

ID = 'id'
TARGET = 'loss'
NFOLDS = 5
SEED = 669
NROWS = None
DATA_DIR = "../../"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test = pd.read_csv(TEST_FILE, nrows=NROWS)

train_indices = train[ID]
test_indices = test[ID]

y_train_full = train["loss"]
y_train_ravel = train[TARGET].ravel()

train.drop([ID, TARGET], axis=1, inplace=True)
test.drop([ID], axis=1, inplace=True)

print("{},{}".format(train.shape, test.shape))

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)


features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    
train = train_test.iloc[:ntrain, :]

# Using train_test_split in order to create random split for Keras,
# otherwise it'll use last part of data when
# validation_split is provided in the model parameters.

X_train, X_val, y_train, y_val = train_test_split(train, y_train_full, test_size = 0.15)

print X_train.shape
print X_val.shape
print y_train.shape
print y_val.shape

x_train_array = np.array(X_train, dtype = float)
y_train_array = np.array(y_train)
x_val_array = np.array(X_val, dtype = float)
y_val_array = np.array(y_val)

# Unfortunately, I didn't manage to implement proper KFold when using Hyperopt.
# This can be done easily using GridSearch.
# Code for 5-fold CV in further section.


# Parameters search space, can be adjusted according to your needs.

space = { 'choice': hp.choice('layers_number',
                             [{'layers': 'two'},
                             {'layers': 'three',
                             'units3': hp.choice('units3', [32, 64, 256]),
                             'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float))
                             }]),

            'units1': hp.choice('units1', [512, 768, 1024]),
            'units2': hp.choice('units2', [128, 256, 512]),
            #'units3': hp.choice('units3', [32, 64, 256]), 

            'dropout1': hp.choice('dropout1', np.linspace(0.3, 0.5, 3, dtype=float)),
            'dropout2': hp.choice('dropout2', np.linspace(0.1, 0.3, 3, dtype=float)),
            #'dropout3': hp.choice('dropout3', np.linspace(0.1, 0.3, 3, dtype=float)),

            'batch_size' : hp.choice('batch_size', [128, 256, 512]),

            'nb_epochs' :  hp.choice('nb_epochs', [30, 50, 100]),
            
        }


# Architecture of NN loosely based on Danijel Kivaranovic Keras script. Thanks!

def neural_net(params):   

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = x_train_array.shape[1], init = 'glorot_normal')) 
    model.add(PReLU())
    model.add(Dropout(0.4))

    model.add(Dense(output_dim=params['units2'], init = "glorot_normal")) 
    model.add(PReLU())
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_normal")) 
        model.add(PReLU())
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss = 'mae', optimizer = 'adam', metrics = ["mae"])
    

    model.fit(x_train_array, y_train_array, nb_epoch=params['nb_epochs'],
              batch_size=params['batch_size'], verbose = 1, validation_data = (x_val_array, y_val_array))

    preds  = model.predict(x_val_array, batch_size = params['batch_size'], verbose = 1)
    acc = mean_absolute_error(y_val_array, preds)
    print('MAE:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}




trials = Trials()
best = fmin(neural_net, space, algo=tpe.suggest, max_evals = 100, trials=trials)
print 'best: '
print best




### GRIDSEARCHCV ###
# Additionally, here implemented with 5 folds.



x_train_array_full = np.array(train, dtype = float)
y_train_array_full = np.array(y_train_full, dtype = float)


def neural_network(size1 = 512, size2 = 256, size3 = 32):
    model = Sequential()
    model.add(Dense(size1, input_dim = x_train_array_full.shape[1], init = 'glorot_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(size2, init = 'glorot_normal'))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(size3, init = 'glorot_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'glorot_normal'))
    model.compile(loss = 'mae', optimizer = 'adam', metrics = ["mae"])
    return(model)

NN_grid = KerasRegressor(build_fn=neural_network, batch_size = 128, nb_epoch = 60, verbose = 1)


print 'Length of data input: ', y_train_array_full.shape[0]



size1 = [256, 512, 1024]
size2 = [64, 128, 256, 512]
size3 = [64, 32, 16, 8]


validator = GridSearchCV(estimator = NN_grid, param_grid = {
        'size1': size1,
        'size2': size2,
        'size3': size3,
    }, cv = 5)
             




grid_result = validator.fit(x_train_array_full, y_train_array_full)

print('The parameters of the best model are: ')
print(validator.best_params_)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

