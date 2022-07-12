# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# preprocessing/decomposition
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA

# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
# It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# supportive models
import xgboost as xgb
# feature selection (from supportive model)
from sklearn.feature_selection import SelectFromModel

# define path to save model
import os
model_path = 'keras_model.h5'

# turn run_fs to True if you want to run the feature selection.
run_fs = False

# turn run_de to True if you want to run the decomposition.
run_de = True

# to make results reproducible
seed = 42

# Read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# save IDs for submission
id_test = test['ID'].copy()

###########################
# DATA PREPARATION
###########################

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False # you can set it = True to ommit multicollinearity (crucial for linear models)
)

print('oh-encoded shape: {}'.format(dummies.shape))

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1), # drop old
        dummies # append them one-hot-encoded
    ],
    axis=1 # column-wise
)

print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train].drop(['ID'], axis=1), total[~is_train].drop(['ID', 'y'], axis=1)

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

#########################
# DECOMPOSITION
#########################
if run_de:
    
    n_comp = 10
    
    train_no_y = train.drop('y', axis=1).copy()
    train_y = train.y
    
    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca_results_train = pca.fit_transform(train_no_y)
    pca_results_test = pca.transform(test)
    
    # FastICA
    ica = FastICA(n_components=n_comp, max_iter=300, random_state=420)
    ica_results_train = ica.fit_transform(train_no_y)
    ica_results_test = ica.transform(test)

    # Append decomposition components to datasets
    for i in range(1, n_comp+1):
        train_no_y['pca_' + str(i)] = pca_results_train[:,i-1]
        test['pca_' + str(i)] = pca_results_test[:, i-1]
   
        train_no_y['ica_' + str(i)] = ica_results_train[:,i-1]
        test['ica_' + str(i)] = ica_results_test[:, i-1]
    
    # Scaling features to between zero and one
    min_max_scaler = MinMaxScaler()
    train_decomposed_no_y = min_max_scaler.fit_transform(train_no_y)
    test_decomposed = min_max_scaler.fit_transform(test)
    print('\nTrain shape after Decomposition: {}\nTest shape after Decomposition: {}'.format(train_decomposed_no_y.shape, test_decomposed.shape))
    print(train_decomposed_no_y)
    
#########################
# FEATURE SELECTION
#########################

if run_fs:
    # XGBoost estimator can be used to compute feature importances,
    # which in turn can be used to discard irrelevant features.
    
    # prepare dict of params for xgboost to run with
    xgb_params = {
        'n_trees': 1000, 
        'eta': 0.005,
        'max_depth': 6,
        'subsample': 0.65,
        'colsample_bytree': 0.95,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'booster':'gbtree',
        'tuneLength': 2
    }
    
    clf = xgb.XGBRegressor(**xgb_params)
    
    train_no_y = train.drop('y', axis=1).copy()
    train_y = train.y
    
    clf = clf.fit(train_no_y, train_y)
    
    features = pd.DataFrame()
    features['feature'] = train_no_y.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    print(features)
    features.plot(kind='barh', figsize=(5, 60))
    
    model = SelectFromModel(clf, prefit=True)
    # no DataFrame
    train_reduced = model.transform(train_no_y)
    # no DataFrame
    test_reduced = model.transform(test)
    print('\nTrain shape after Feature Selection: {}\nTest shape after Feature Selection: {}'.format(train_reduced.shape, test_reduced.shape))

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
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
# Base model architecture definition.
# Dropout is a technique where randomly selected neurons are ignored during training. 
# They are dropped-out randomly. This means that their contribution to the activation.
# of downstream neurons is temporally removed on the forward pass and any weight updates are
# not applied to the neuron on the backward pass.
# More info on Dropout here http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# BatchNormalization, Normalize the activations of the previous layer at each batch, i.e. applies a transformation 
# that maintains the mean activation close to 0 and the activation standard deviation close to 1.
# 'input_dims' inputs -> ['input_dims' -> 'input_dims' -> 'input_dims'//2 -> 'input_dims'//4] -> 1
def model():
    model = Sequential()
    #input layer
    model.add(Dense(input_dims, input_dim=input_dims, kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25)) # Reduce Overfitting With Dropout Regularization
    # hidden layers
    model.add(Dense(input_dims,kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.25))
    
    model.add(Dense(input_dims//2,kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.4))
    
    model.add(Dense(input_dims//4))
    model.add(Activation(act_func))
  
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    # Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer=adam,
                  metrics=[r2_keras] # you can add several if needed
                 )
    
    # Visualize NN architecture
    print(model.summary())
    return model
    
# initialize input dimension

if run_fs:
    # with Feaure Selection
    input_dims = train_reduced.shape[1]
elif run_de:
    # Decomposition
    input_dims = train_decomposed_no_y.shape[1]
else:
    # No Feature Selection
    input_dims = train.shape[1]-1
    
#activation functions for hidden layers
act_func = 'sigmoid' # could be 'relu', 'sigmoid', ...tanh

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=500, 
    batch_size=10,
    verbose=1
)

# X, y preparation
if run_fs:
     # With Feature Selection
     X, y = train_reduced, train_y
     X_test = test_reduced
     print('\nTrain shape after Feature Selection: {}\nTest shape after Feature Selection: {}'.format(X.shape, X_test.shape))

elif run_de:
     # Decomposition
     X, y = train_decomposed_no_y, train_y
     X_test = test_decomposed
     print('\nTrain shape Decomposition: {}\nTest shape Decomposition: {}'.format(X.shape, X_test.shape))
else:    
     # No Feature Selection
     X, y = train.drop('y', axis=1).values, train.y.values
     X_test = test.values
     print('\nTrain shape No Feature Selection: {}\nTest shape No Feature Selection: {}'.format(X.shape, X_test.shape))

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=seed
)

# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=50,
        verbose=1),
    
    ModelCheckpoint(
        model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=0)
]

# fit estimator
estimator.fit(
    X_tr, 
    y_tr, 
    epochs=500, # increase it to 20-100 to get better results
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y_tr, estimator.predict(X_tr))**0.5)) # mse train
print('R^2 train: {}'.format(r2_score(y_tr, estimator.predict(X_tr)))) # R^2 train

# check performance on validation set
print('MSE val: {}'.format(mean_squared_error(y_val, estimator.predict(X_val))**0.5)) # mse val
print('R^2 val: {}'.format(r2_score(y_val, estimator.predict(X_val)))) # R^2 val
pass

# predict results
res = estimator.predict(X_test).ravel()

# create df and convert it to csv
output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv('keras-baseline.csv', index=False)