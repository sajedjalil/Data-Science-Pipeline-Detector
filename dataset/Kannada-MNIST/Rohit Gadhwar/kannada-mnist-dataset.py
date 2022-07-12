# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
test = test.iloc[:, 1:].values

X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_train = X_train.reshape(X_train.shape[0], 28, 28)/255.0
X_train = X_train.reshape(X_train.shape[0], 28,28,1)
test = test.reshape(test.shape[0], 28, 28)/255.0
test = test.reshape(test.shape[0], 28,28,1)

# Importing the Keras libraries and packages
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Conv2D(16, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

classifier.add(Conv2D(32, kernel_size = 3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, kernel_size = 3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, kernel_size = 3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, kernel_size = 3, padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation='softmax'))

# Compiling the ANN
optmize = tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
classifier.compile(loss = 'sparse_categorical_crossentropy', optimizer = optmize,  metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 128, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(test)
results = np.argmax(y_pred,axis = 1)

data_out = pd.DataFrame({'id': range(len(test)), 'label': results})
data_out.to_csv('submission.csv', index = None)