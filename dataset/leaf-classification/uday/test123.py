# import required libraries 
# %pylab inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential, Merge
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# load data 
# data = pd.read_csv('../input/train.csv')
data = pd.DataFrame.from_csv('../input/train.csv')

# print(data.head())

y = data['species']
y = LabelEncoder().fit(y).transform(y)
y_cat = to_categorical(y)

margin = data.columns[1:65]
shape = data.columns[65:129]
texture = data.columns[129:193]
# print(margin)
# print(shape)
# print(texture)

modelMargin = Sequential()
modelMargin.add(Dense(128, input_dim=64, activation='relu'))
modelMargin.add(Dropout(0.3))

modelShape = Sequential()
modelShape.add(Dense(128, input_dim=64, activation='relu'))
modelShape.add(Dropout(0.3))

modelTexture = Sequential()
modelTexture.add(Dense(128, input_dim=64, activation='relu'))
modelTexture.add(Dropout(0.3))

merged = Merge([modelMargin, modelShape, modelTexture], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(99, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    [data[margin].as_matrix(), data[shape].as_matrix(), data[texture].as_matrix()], 
    y_cat, 
    nb_epoch=350,
    batch_size=64,
    validation_split=0.3
)              

print()
print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

## read test file
# test1 = pd.read_csv('../input/test.csv')
# test = pd.DataFrame.from_csv('../input/test.csv')
# index = test1.pop('id')
# # index = test['id']

# testMargin = test.columns[0:64]#.as_matrix()
# testShape = test.columns[64:128]#.as_matrix()
# testTexture = test.columns[128:192]#.as_matrix()

# # testMargin = StandardScaler().fit(testMargin).transform(testMargin)
# # testShape = StandardScaler().fit(testShape).transform(testShape)
# # testTexture = StandardScaler().fit(testTexture).transform(testTexture)

# yPred = model.predict_proba([test[testMargin].as_matrix(), test[testShape].as_matrix(), test[testTexture].as_matrix()])

# ## Converting the test predictions in a dataframe as depicted by sample submission
# yPred = pd.DataFrame(yPred, index=index, columns=data['species'].unique())
# fp = open('submission_nn_kernel.csv','w')
# fp.write(yPred.to_csv())






# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "."]).decode("utf8"))

# Any results you write to the current directory are saved as output.