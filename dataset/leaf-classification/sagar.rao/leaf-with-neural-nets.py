##  libraries


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

##Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

## Read data from the CSV file

train = pd.read_csv('../input/train.csv')
parent_data = train.copy()   
index = train.pop('id')

labels = train.pop('species')
labels = LabelEncoder().fit(labels).transform(labels)
print(labels.shape)

X = preprocessing.MinMaxScaler().fit(train).transform(train)
trainn = StandardScaler().fit(X).transform(X)
print(trainn.shape)

labels = to_categorical(labels)
print(labels.shape)
####stratified splitting
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.2, random_state=12345)
sss.get_n_splits(trainn, labels)
print(sss)
for train_index, test_index in sss.split(trainn, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index] 
    
    
model = Sequential()
model.add(Dense(768,input_dim=192,  init='glorot_normal', activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(768, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(99, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=300)
history = model.fit(X_train, y_train,batch_size=192,epochs=2500 ,verbose=0,
                    validation_data=(X_test, y_test),callbacks=[early_stopping])

print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

test = pd.read_csv('../input/test.csv')
index = test.pop('id')
test= preprocessing.MinMaxScaler().fit(test).transform(test)
test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)

yPred = pd.DataFrame(yPred,index=index,columns=sorted(parent_data.species.unique()))

fp = open('submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())






























