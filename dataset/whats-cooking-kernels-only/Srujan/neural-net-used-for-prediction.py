# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.svm import SVC
import pandas as pd
import json
import os
os.listdir("../input")
os.listdir("../input")
train = json.load(open(('../input/train.json')))
test = json.load(open(('../input/test.json')))

def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data

train_X = generate_text(train)
test_X = generate_text(test)
train_y = [doc['cuisine'] for doc in train]
tfdif = TfidfVectorizer()

train_X_transformed = tfdif.fit_transform(train_X).toarray()
test_X_transformed = tfdif.transform(test_X).toarray()

lb = LabelEncoder()
train_y_encoded = lb.fit_transform(train_y)
train_y_encoded = train_y_encoded.reshape(-1, 1)
onehotencoder = OneHotEncoder(categories='auto')
train_y_encoded = onehotencoder.fit_transform(train_y_encoded).toarray()


inputX = train_X_transformed
inputY = train_y_encoded



from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=181, kernel_initializer='uniform', activation='relu', input_dim=inputX.shape[1]))
classifier.add(Dropout(0.2))
# Adding the second hidden layer
classifier.add(Dense(units=90, kernel_initializer='uniform', activation='relu'))
# Adding the third hidden layer
# classifier.add(Dense(units=90, kernel_initializer='uniform', activation='relu'))

classifier.add(Dropout(0.2))
# Adding the output layer
classifier.add(Dense(units=inputY.shape[1], kernel_initializer='uniform', activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # both 'Adagrad' and 'adam' works good for our dataset(optimizer='RMSprop' is good if we are using recurrent neural networks.)
# Adagrad is an optimizer with parameter-specific learning rates,

# Reduces learning rate when there is no improvement in learning rate
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
# Fitting the ANN to the Training set.
classifier.fit(inputX, inputY, batch_size=500, epochs=30, callbacks=[reduce_lr])

y_pred = classifier.predict(test_X_transformed)
print(y_pred)
y_pred = onehotencoder.inverse_transform(y_pred)
y_pred = lb.inverse_transform(y_pred)


test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('submission.csv', index=False)