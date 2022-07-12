import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical

data = pd.read_csv('../input/train.csv')
parent_data = data.copy() 
ID = data.pop('id')

y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
X = StandardScaler().fit(data).transform(data)
y_cat = to_categorical(y)

model = Sequential()
model.add(Dense(1024,input_dim=192))
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Activation('sigmoid'))
model.add(Dense(99))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

history = model.fit(X,y_cat,batch_size=128,nb_epoch=120,verbose=0)
test = pd.read_csv('../input/test.csv')

index = test.pop('id')

test = StandardScaler().fit(test).transform(test)

ypred = model.predict_proba(test)

ypred = pd.DataFrame(ypred,index=index,columns=parent_data.species.unique())
fp = open('submission.csv','w')
fp.write(ypred.to_csv())



