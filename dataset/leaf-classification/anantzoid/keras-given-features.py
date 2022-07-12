# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def prepare_data(train,  test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    labels_cat = to_categorical(labels)
    classes = list(le.classes_)

    test_ids = test.id
    train_ids = train.id

    train = train.drop(['id', 'species'], axis=1)
    test = test.drop(['id'], axis=1)

    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, labels_cat, classes, test_ids, test

def make_submit(preds):
    submission = pd.DataFrame(preds, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()
    submission.to_csv('submit.csv', index=False)

train, labels, classes, test_ids, test = prepare_data(train, test)


model = Sequential()
model.add(Dense(512, input_dim=192, init="glorot_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(1024, init="glorot_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1024, init="glorot_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(512, init="glorot_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dense(99, init="glorot_normal"))
model.add(Activation("softmax"))
#epochs, lr = 800, 1e-3
#decay = lr/epochs
adam = Adam()#learning_rate=lr, decay=decay)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
# Run with more epochs on local server
model.fit(train, labels, nb_epoch=200, batch_size=128)

preds = model.predict_proba(test)
make_submit(preds)



