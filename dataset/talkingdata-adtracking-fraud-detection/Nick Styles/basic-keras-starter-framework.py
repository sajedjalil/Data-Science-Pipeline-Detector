# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import extra libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.utils import class_weight # attempt to manage the class unbalance

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Read in the train file, selecting 30 million records

data_in = pd.read_csv('../input/train.csv', nrows=3000000)

data_in.info()



# Specify the variables, since there will be no feature engineeering we'll select a few only

x = data_in.ix[:,:5]



# Specify the lable

data_in.is_attributed = float(data_in.is_attributed)

y = np.ravel(float(data_in.is_attributed))



# Split the train set for training and validation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)



# Define the scalar and scale the training and validation data

scaler = StandardScaler().fit(x_train)



x_train = scaler.transform(x_train)



x_test = scaler.transform(x_test)



# Select the model to use

model = Sequential()



# Define the various input layers

# Input layer

# Since we selected 5 variables for training: input_shape = 5 and for the node = 5 + 1 for bias

model.add(Dense(6, activation='relu', input_shape=(5,)))



# Hidden layer

# The value is preferably somewhere between the input and output layer. 4 was arbitrarily chosen.

model.add(Dense(4, activation='relu'))



# Output layer

# Since we using a classification model and not dealing with multiclass, 1 node is sufficient.

model.add(Dense(1, activation='sigmoid'))



# Compile the model

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

              

# Create weights based on the lable           

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)



# Train the model

# Epoch = 1, change this depending on the loss and the accuracy values

# Increase batch size to speed up large amounts of records used for training

model.fit(x_train, y_train,epochs=5, batch_size=512, verbose=1, validation_data=(x_test,y_test), shuffle=True, class_weight=class_weights)



# Test the model

y_pred = model.predict(x_test)



# Evaluate the model

score = model.evaluate(x_test, y_test,verbose=1)

print(score) # Loss first and the accuracy seecond



# Some checks to see how the model performed

confusion_matrix(y_test, y_pred.round())

precision_score(y_test, y_pred.round())

recall_score(y_test, y_pred.round())



# Read in the test file

test = pd.read_csv('../input/test.csv')



# Select the same variables used in the testing

test_ix  =test.ix[:,1:6]



# Predict the test file for submission

prediction=model.predict_classes(test_ix)



output = pd.DataFrame(data=prediction)

output.head()

# Rename columns

output.columns = ['is_attributed']

output['click_id'] = output.index



# Create csv for submission

# The submission below scores around 50%

output.to_csv('submission.csv', index=False)






