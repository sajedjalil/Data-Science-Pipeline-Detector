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
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adam

np.random.seed(1)  # for reproducibility

print("Loading data...")
train_in = pd.read_csv('../input/train.csv')
test_in = pd.read_csv('../input/test.csv')

print("Creating dummy variables...")
y_train = train_in.iloc[:, -1]
data = pd.concat((train_in.iloc[:, :-1], test_in))
# Get dummy variables for all categorical columns
colnames = []
X_data = []
for c in [i for i in data.columns[1:-1] if 'cat' in i]:
    add = pd.get_dummies(data[c]).iloc[:,:-1].values.astype(np.bool)
    X_data.append(add)
    colnames += [c + '_' + str(i) for i in range(add.shape[1])]
X_data = pd.DataFrame(np.hstack(X_data), columns=colnames)
X_data = X_data.iloc[:, [len(pd.unique(X_data.loc[:,c]))>1 for c in X_data.columns]]
X_data_cont = np.vstack([data[c].values.astype(np.float32) \
                         for c in data.columns[1:-1] if 'cat' not in c]).T
X_data = X_data.join(pd.DataFrame(X_data_cont, 
                    columns=[c for c in data.columns[1:-1] if 'cat' not in c]))
# Create train and test tables
train = X_data[:len(y_train)].join(y_train)
test = X_data[len(y_train):]

def build_model(in_dim, dropout_p=0.5, l2_reg=0.001):                                
    model = Sequential()
    model.add(Dropout(0.1, input_shape=[in_dim]))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mae', optimizer=Adam())
    return model

Xt = train.iloc[:, :-1].values
yt = train.iloc[:, -1].values

minny = min(yt)
print(minny)

if minny > 0:
    yt = np.log(train.iloc[:, -1].values)
else:
    yt = np.log1p(train.iloc[:, -1].values)
    
print("Compiling model...")
model = build_model(in_dim=Xt.shape[1])
model.compile(loss='mae', optimizer=Adam())
    
print("Fitting model...")
history = model.fit(Xt, yt,
                    batch_size=96,
                    nb_epoch=1,
                    verbose=2, 
                    #validation_data=(Xv.values, yv.values),
                    shuffle=True)
                    
preds = model.predict(test.values)
if minny > 0:
    preds = np.exp(preds)
else:
    preds = np.expm1(preds)
submission = pd.DataFrame(test_in['id'])
submission['loss'] = preds
submission.to_csv('submission_nn.csv', index=False)