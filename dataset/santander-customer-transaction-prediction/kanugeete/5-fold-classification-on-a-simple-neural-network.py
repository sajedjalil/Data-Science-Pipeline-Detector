# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
from sklearn.model_selection import KFold
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
traincols = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']
folds = KFold(n_splits=5, random_state=None)
model = Sequential()
model.add(Dense(200,activation='relu',input_shape=(200,),kernel_initializer='glorot_uniform'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

for train_index, test_index in folds.split(train):
    #print(“TRAIN:”, train_index, “TEST:”, test_index)
    X_train, X_valid = train.iloc[train_index], train.iloc[test_index]
    y_train, y_valid = target[train_index], target[test_index]
    X_train = X_train[traincols]
    X_valid = X_valid[traincols]
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train,y_train,batch_size=256,epochs=5,validation_data=(X_valid,y_valid))

predictions = model.predict(test[traincols])
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv('msubmission.csv', index=False)