# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import library
import pandas as pd


# import dataset
dataset = pd.read_csv('../input/train.csv')

dataset.set_index('ID_code')


y = dataset.iloc[:, 1]
X = dataset.iloc[:, 2:]


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(200, activation='relu', kernel_initializer='uniform', input_dim=200))

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, batch_size=128, epochs=50, verbose=2, validation_split=0.2, shuffle=True)

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])  # RAISE ERROR
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #RAISE ERROR
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_dataset = pd.read_csv('../input/test.csv')

y_check = test_dataset.iloc[:, 1:]

result = model.predict(y_check, verbose=1)

out = []

for i in range(0, result.shape[0]):
    out.append(result[i][0])

output = pd.DataFrame({'ID_code': test_dataset['ID_code'], 'target': list(out)})

output.to_csv('result.csv', index=False)