# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Merge, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

print("Load Training Data...")
train = pd.read_csv("../input/train.csv")
X_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
Y_train = le.transform(train['species'])

print("Load Test Data...")
test = pd.read_csv("../input/test.csv")
test_id = test['id']
X_test = test.drop(['id'], axis=1).values


"Standardized the data with unit variance and unit mean"
"Use this scalar to standardize the test data also"
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print(X_train.shape)
print(X_test.shape)
input_dim = X_train.shape[1] / 3

X_margin_train = X_train[:, input_dim*0:input_dim*1]
X_shape_train = X_train[:, input_dim:input_dim*2]
X_texture_train = X_train[:, input_dim*2:input_dim*3]

X_margin_test = X_test[:, input_dim*0:input_dim*1]
X_shape_test = X_test[:, input_dim:input_dim*2]
X_texture_test = X_test[:, input_dim*2:input_dim*3]


print("Defining the NN model...")

margin_model = Sequential()
margin_model.add(Dense(output_dim=input_dim, input_dim=input_dim))

shape_model = Sequential()
shape_model.add(Dense(output_dim=input_dim, input_dim=input_dim))

texture_model = Sequential()
texture_model.add(Dense(output_dim=input_dim, input_dim=input_dim))


model = Sequential()
model.add(Merge([margin_model, shape_model, texture_model], mode='concat', concat_axis=1))
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
model.fit([X_margin_train, X_shape_train, X_texture_train], Y_train, nb_epoch=100, batch_size=1, verbose=1)

print("Prediction \n")
Y_test = model.predict_proba([X_margin_test, X_shape_test, X_texture_test])

print(Y_test[1])

submission = pd.DataFrame(Y_test, index=test_id, columns=le.classes_)
submission.to_csv('submission_NN.csv')