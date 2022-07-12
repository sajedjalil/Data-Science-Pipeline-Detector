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

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('../input/train/train.csv')
testdata = pd.read_csv('../input/test/test.csv')

X_train = data.drop(columns=['Name','RescuerID','Description','PetID',
                             'AdoptionSpeed']).values
y_train = data.iloc[:, 23].values

X_test = testdata.drop(columns=['Name','RescuerID','Description','PetID']).values

# Stack to get same dimensions of testing and training
stack = np.vstack((X_train,X_test))

for i in range(len(stack[:, 0])):
    if stack[i, 2] == stack[i, 3]:
        stack[i, 3] = 0

# Categorical Data
onehotencoderx = OneHotEncoder(categorical_features = [2,3,5,6,7,10,11,12,16])
stack = onehotencoderx.fit_transform(stack).toarray()
onehotencodery = OneHotEncoder(sparse = False)
y_train = y_train.reshape(len(y_train), 1)
y_train = onehotencodery.fit_transform(y_train)

# Split stack back into training and testing
train_len = len(X_train[:,0])
X_train = stack[0:train_len,:]
stack_len = len(stack[:,0])
X_test = stack[train_len:(stack_len+1),:]

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Run a ANN on the data
len_input = len(X_train[0,:])

classifier = Sequential()
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', 
                     input_dim = len_input))
classifier.add(Dense(output_dim = 150, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 200, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])
classifier.fit(X_train, y_train, nb_epoch = 50)
        
y_pred = classifier.predict(X_test)


final_preds = []

for i in range(len(y_pred[:,0])):
    temp = list(y_pred[i,:])
    cl = temp.index(max(temp))
    final_preds.append(cl)

IDs = list(testdata.iloc[:,21])

final_frame = pd.DataFrame(
    {'PetID': IDs,
     'AdoptionSpeed': final_preds
    })

final_frame.to_csv('submission.csv', index = False)