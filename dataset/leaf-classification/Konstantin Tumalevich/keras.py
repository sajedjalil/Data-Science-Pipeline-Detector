import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical


data = pd.read_csv('../input/train.csv')
orig_data = data.copy()
ids = data.pop('id')
data_y = data.pop('species')
data_y_encoded = LabelEncoder().fit(data_y).transform(data_y)
data_x = StandardScaler().fit(data).transform(data)
y_cat = to_categorical(data_y_encoded)


model = Sequential()
model.add(Dense(1024,input_dim=192,  init='uniform', activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(99, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])

train_iterations = model.fit(data_x, y_cat, batch_size=128, nb_epoch=100, verbose=0, validation_split=0.1)

#print(min(train_iterations.history['val_acc']))

#relu - 0.464646458626
#tanh - 0.494949489832
#sigmoid - 0.0404040403664
#hard_sigmoid - 0.0101010100916
#linear - 0.54545456171

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
test_x = StandardScaler().fit(test).transform(test)

test_y = model.predict_proba(test_x)

submission = pd.DataFrame(test_y,index=test_ids,columns=sorted(orig_data.species.unique()))
submission.to_csv('submission11.csv')



