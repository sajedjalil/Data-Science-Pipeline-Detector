#santander-customer-transaction-prediction
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
from keras import losses
import tensorflow as tf
from keras import backend as K

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

# check for missing value
train.isnull().sum().sum()
test.isnull().sum().sum()

#Coulmn Names
col_m =[c for c in train.columns if c not in ['ID_code','target']]
  
#train_set=train_sel.values
Y = train['target']
X = train[col_m]
X_test = test[col_m]

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.25, random_state = 50)

#Define AUC
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#Define Neural Net Model
model = Sequential()
model.add(Dense(200,activation='relu', input_dim=200, kernel_initializer='normal'))
model.add(Dense(100,activation='relu', kernel_initializer='normal'))
model.add(Dense(50,activation='tanh', kernel_initializer='normal'))
model.add(Dense(10,activation='tanh', kernel_initializer='normal'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=losses.binary_crossentropy, optimizer='sgd', metrics=['accuracy', auc])
model.summary()

model.fit(X_train.values, Y_train.values, validation_data=(X_valid.values, Y_valid.values),nb_epoch=10, batch_size=16,verbose=0)

test_pred = model.predict(X_test.values)

submission=test[['ID_code']]
submission['target']= test_pred
submission.to_csv('keras_submission_20190222.csv',index=False)









