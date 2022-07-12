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

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils

# Data
df = pd.read_csv('../input/train.csv')
df.head()
X_train = df.iloc[:, 1:-1].values
y_train = df.iloc[:, -1].values
y_train = np_utils.to_categorical(y_train)

# Model
model = Sequential()
model.add(Dense(2048, input_dim=54,kernel_initializer='uniform', activation='relu'))
model.add(Dense(1024, kernel_initializer='uniform', activation='relu'))
model.add(Dense(512, kernel_initializer='uniform', activation='relu'))
model.add(Dense(256, kernel_initializer='uniform', activation='softplus'))
model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='uniform', activation='softplus'))
model.add(Dense(8, kernel_initializer='uniform', activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X_train, y_train, epochs=250, batch_size=32)

#Predict
df_test = pd.read_csv('../input/test.csv')
X_test = df_test.iloc[:, 1:].values
preds = model.predict(X_test)
sub = pd.DataFrame({"Id": df_test.iloc[:, 0].values, "Cover_Type": np.argmax(preds, axis=1)})
sub.to_csv("etc.csv", index=False)
