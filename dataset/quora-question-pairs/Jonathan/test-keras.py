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


import sys
import re
import os
import time
import string

from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.utils import np_utils


import matplotlib.pyplot as plt

# This function transforms a string to a onehot encoded vectors,
# e.g. "abc" -> [1, 2, 3] -> [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
def string_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng.lower()]
    return vector


train = pd.read_csv("../input/train.csv",nrows=20)
#test  = pd.read_csv("../input/test.csv",nrows=200)

q1_img = train["question1"].apply(lambda x : string_vectorizer(str(x))).as_matrix()
q2_img = train["question2"].apply(lambda x : string_vectorizer(str(x))).as_matrix()



from keras.preprocessing.sequence import pad_sequences
q1_data = pad_sequences(q1_img, maxlen=50)
q2_data = pad_sequences(q2_img, maxlen=50)

plt.figure()
plt.title(train.iloc[0]["question1"])
plt.imshow(string_vectorizer(train.iloc[0]["question1"]))
plt.show()
plt.savefig("sent.png")

labels = np.array(train["is_duplicate"], dtype=int)

print('train shape:', q1_data.shape)
print(q1_data.shape[0], 'train samples')


sys.exit()

model_q1 = Sequential()
model_q1.add(Conv1D(10, 10, border_mode='same',
                    input_shape=q1_data.shape[1:],
                    activation='relu'))
model_q1.add(MaxPooling1D(5))
model_q1.add(Conv1D(10, 10, border_mode='same',
                    input_shape=q1_data.shape[1:],
                    activation='relu'))
model_q1.add(MaxPooling1D(10))
model_q1.add(Flatten())



model_q2 = Sequential()
model_q2.add(Conv1D(10, 10, border_mode='same',
                    input_shape=q1_data.shape[1:],
                    activation='relu'))
model_q2.add(MaxPooling1D(5))
model_q2.add(Conv1D(10, 10, border_mode='same',
                    input_shape=q1_data.shape[1:],
                    activation='relu'))
model_q2.add(MaxPooling1D(10))
model_q2.add(Flatten())


model = Sequential()
model.add(Merge([model_q1, model_q2], mode='concat'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam')


history = model.fit([q1_data,q2_data],labels,
          batch_size=100,
          nb_epoch=20,
          validation_split=0.2)


plt.figure()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
#plt.savefig("train.png")

