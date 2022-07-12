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


# Import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
id_column_subm = test.iloc[:,0].values

X = train.drop(labels = ['Cover_Type'], axis = 1)
y_raw = train.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y_raw.reshape(-1,1)).toarray()

X = X.drop(labels = ['Id','Soil_Type40', 'Wilderness_Area4'], axis = 1)
test = test.drop(labels = ['Id','Soil_Type40','Wilderness_Area4'], axis = 1)

X = X.values
test = test.values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
test = sc_X.transform(test)

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 50, kernel_initializer='uniform', activation = 'relu', input_dim = 52))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 30, kernel_initializer= 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the third hidden layer
classifier.add(Dense(units = 15, kernel_initializer= 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 7, kernel_initializer= 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, epochs = 160)

#Predicting values for classes [classes from 1-7, values predicted from 0-6 so add 1]
y_pred = classifier.predict_classes(test) + 1


submission = pd.DataFrame({
        "Id": id_column_subm,
        "Cover_Type": y_pred
    })
submission.to_csv('submission.csv', index=False)
