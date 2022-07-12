# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy# linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense,Dropout,core
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#load train dataset
dataframe = pandas.read_csv('../input/train.csv')
dataset = dataframe.values
X = dataset[:,2:].astype(float)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = dataset[:,1]
encoder = LabelEncoder()
le=encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#convert integers to dummy variables 
dummy_y = np_utils.to_categorical(encoded_Y)

#load test dataset
test = pandas.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

# create model
model = Sequential()
model.add(Dense(384, input_dim=192, init='uniform', activation='relu'))
model.add(Dense(384, init='uniform', activation='relu'))
model.add(Dense(99, init='uniform', activation='softmax'))

# Compile and fit model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, nb_epoch=200, batch_size=20)

#make predictions and submission
predictions=model.predict_proba(x_test)
submission = pandas.DataFrame(predictions, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
