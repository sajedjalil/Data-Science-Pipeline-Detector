# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

# Features are based on Andrew Lukyanenko's kernel at https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples
Xtrain = pd.read_csv('../input/lanl-competition-fe-v1/x_train')
Ytrain = pd.read_csv('../input/lanl-competition-fe-v1/y_train')
Xtest = pd.read_csv('../input/lanl-competition-fe-v1/x_test')


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

X_train, X_val, Y_train, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=5)

#Libraries for neural net
import keras
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

#Model parameters

kernel_init = 'he_normal'
input_size = len(Xtrain.columns)

### Neural Network ###

# Model architecture: A very simple shallow Neural Network 
model = Sequential()
model.add(Dense(16, input_dim = input_size)) 
model.add(Activation('linear'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(32))    
model.add(Activation('tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))    
model.add(Activation('linear'))

#compile the model
optim = optimizers.Adam(lr = 0.001)
model.compile(loss = 'mean_absolute_error', optimizer = optim)

#Callbacks
csv_logger = CSVLogger('log.csv', append=True, separator=';')
best_model = ModelCheckpoint("model.hdf5", save_best_only=True, period=3)
restore_best = EarlyStopping(monitor='val_loss', verbose=2, patience=100, restore_best_weights=True)

model.fit(x=X_train, y=Y_train, batch_size=64, epochs=200, verbose=2, callbacks=[csv_logger, best_model], validation_data=(X_val,Y_val))
### Neural Network End ###

nn_predictions = model.predict(Xtest, verbose = 2, batch_size = 64)
submission['time_to_failure'] = nn_predictions
submission.to_csv('submission.csv')























