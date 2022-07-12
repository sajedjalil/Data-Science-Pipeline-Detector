# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

"Load Training Data"
train = pd.read_csv('../input/train.csv')
X_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
Y_train = le.transform(train['species'])


"Load Test Data"
test = pd.read_csv('../input/test.csv')
test_id = test['id']
X_test = test.drop(['id'], axis=1).values


"Standardized the data with unit variance and unit mean"
"Use this scalar to standardize the test data also"
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)
input_dim = X_train.shape[1]

print("Defining the NN model")
model = Sequential()
model.add(Dense(output_dim=input_dim, input_dim=input_dim))
model.add(Activation('linear'))
model.add(Dense(120))
model.add(Dropout(0.2))
model.add(Dense(120))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(99))
model.add(Activation('sigmoid'))
model.add(Dense(99))
model.add(Activation('softmax'))
print("Compiling model")
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
print("Fit the model to data")
model.fit(X_train, Y_train, nb_epoch=20, batch_size=1, verbose=1)

print("Prediction \n")
Y_test = model.predict_proba(X_test)

print(Y_test[1])

submission = pd.DataFrame(Y_test, index=test_id, columns=le.classes_)
submission.to_csv('submission_NN.csv')
