import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import keras as k
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout

# read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train["y"]
train = train.drop(['y'], axis=1)

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

##Add decomposed components: PCA / ICA etc.
n_comp = 12

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

X_train, X_val, y_train2, y_val = train_test_split(train, y_train, test_size=0.15, random_state=42)
size = X_train.shape[1]

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=size))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])
      
model.fit(X_train.as_matrix(), y_train2.as_matrix(),
          batch_size=64,
          epochs=200,
          verbose=1,
          validation_data=(X_val.as_matrix(), y_val.as_matrix()))

y_pred = model.predict(test.as_matrix())

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred.flatten()})
output.to_csv('submission_baseLine.csv', index=False)
