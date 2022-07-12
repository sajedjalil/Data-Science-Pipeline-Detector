
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#uploading the data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
target = pd.read_csv('../input/sample_submission.csv')

#Spliting the data into train and test sets

X = train_data.iloc[:,2:202].values
y = train_data.iloc[:,1].values


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 99999)


##Creating an Object of StandardSacaler
#Standard Skale for Train data
obj_standart_train = StandardScaler()
X_train = obj_standart_train.fit_transform(X_train)

#Standard Skale for Test data
obj_standart_test = StandardScaler()
X_test = obj_standart_test.fit_transform(X_test)
################################################

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu', input_dim = 200))

# Adding the second hidden layer
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'softmax' , input_dim = 200))


# Adding the second hidden layer
classifier.add(Dense(units = 510, kernel_initializer = 'uniform', activation = 'softmax'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
classifier.save('classifier.h5')
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.85)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)


######################################################################
#predicting test data
# Splitting the dataset into the Training set and Test set

X_t = test_data.iloc[:,1:202]

t_target=target.iloc[:,1]


##Creating an Object of StandardSacaler
#Standard Skale for Train data
obj_standart_test = StandardScaler()
X_t = obj_standart_test.fit_transform(X_t)


# Predicting the Test set results
y_pred_t = classifier.predict(X_t)
#y_pred_t = (y_pred_t > 0.7)

target['target'] = y_pred_t
#target.drop(['target'],axis=1, inplace = True)

target.to_csv('submission.csv', index=False)