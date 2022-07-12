 # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm import tqdm
import cv2
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = preprocessing.LabelEncoder()

path = "../input/humpback-whale-identification/train"

train = pd.read_csv("../input/humpback-whale-identification/train.csv")
train_label = np.array(train["Id"])
train_label = le.fit_transform(train_label)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = train_label.reshape(len(train_label), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

y = onehot_encoded
training_data = []

"""for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        new_array = cv2.resize(img_array, (100, 100))  # resize to normalize data size
        training_data.append([new_array])  # add this to our training_data


random.shuffle(training_data)
X = []

for features in training_data:
    X.append(features)

X = np.array(X).reshape(-1, 100, 100, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()"""

pickle_in = open("../input/example/X.pickle","rb")
X = pickle.load(pickle_in)
X = X/255

model = Sequential()


model.add(Conv2D(64, (5,5),padding='same',input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(y.shape[1]))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X, y,batch_size=32,epochs=8,validation_split=0.3)



