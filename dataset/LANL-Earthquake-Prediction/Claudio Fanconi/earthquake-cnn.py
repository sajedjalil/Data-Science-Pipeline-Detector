'''
Earthquake time predictor using various mathematical prediction models
Author: fanconic
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.precision = 15
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings("ignore")

N_ROWS = 5e6
train = pd.read_csv('../input/train.csv',        
                #nrows= N_ROWS, 
                dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print('Train input completed')

X_train = train['acoustic_data']
y_train = train['time_to_failure']

del train

# Cut training data
rows = 150_000
X_train = X_train[:int(np.floor(X_train.shape[0] / rows))*rows]
y_train = y_train[:int(np.floor(y_train.shape[0] / rows))*rows]
X_train= X_train.values.reshape((-1, rows, 1))
print(X_train.shape)

y_train = y_train[rows-1::rows].values
print(y_train.shape)

# Training/ Vaidation Split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                y_train,
                                                test_size= 0.2,
                                                random_state= 11)

# Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()

# Conv 1
model.add(Conv1D(32, 10, activation='relu', input_shape=(rows, 1)))

# Max Pooling
model.add(MaxPooling1D(100))

# Conv 3
model.add(Conv1D(64, 10, activation='relu'))

# Average Pooling
model.add(GlobalAveragePooling1D())

model.add(Dense(16, kernel_initializer='normal',activation='relu'))
# Output Layer
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              verbose=0,
                              mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
                           save_best_only=True,
                           monitor='val_loss',
                           mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')

model.compile(loss='mean_absolute_error', optimizer= adam(lr=1e-4), metrics=['mean_absolute_error'])
model.summary()

model.fit(X_train, 
        y_train,
        batch_size= 16,
        epochs= 100, 
        validation_data= (X_val, y_val),
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        verbose= 0
         )


y_pred = model.predict(X_val)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show()

# training Score
score = mean_absolute_error(y_val.flatten(), y_pred)
print(f'Score: {score:0.3f}')

# Submission
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = []

for segment in tqdm(submission.index):
        seg = pd.read_csv('../input/test/' + segment + '.csv')
        x = pd.Series(seg['acoustic_data'].values)
        X_test.append(x)

X_test = np.asarray(X_test)
X_test = X_test.reshape((-1, 1))
print(X_test.shape)
X_test = X_test[:int(np.floor(X_test.shape[0] / rows))*rows]
X_test= X_test.reshape((-1, rows, 1))
print(X_test.shape)
submission['time_to_failure'] = model.predict(X_test)
submission.to_csv('submission.csv')