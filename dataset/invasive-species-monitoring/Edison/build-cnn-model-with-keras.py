# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_labels_f = pd.read_csv('../input/train_labels.csv')
sample_f = pd.read_csv('../input/sample_submission.csv')

path = '../input/train/'
y = []
img_path = []
for i in range(len(train_labels_f)):
    img_path.append(path + str(train_labels_f.loc[i][0]) + '.jpg')
    y.append(train_labels_f.loc[i][1])
y = np.array(y)

import cv2
import matplotlib.pyplot as plt
import random

x = []
img_size = 128
for i, file_name in enumerate (img_path):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (img_size, img_size))
    x.append(img)
x = np.array(x)

index_va = random.sample(range(0,len(y)),int(len(y)*0.3))
test_x = []
train_x = []
test_y = []
train_y = []
for i in range(len(y)):
    if i in index_va:
        test_x.append(x[i])
        test_y.append(y[i])
    else:
        train_x.append(x[i])
        train_y.append(y[i])
test_x = np.array(test_x)
test_y = np.array(test_y)
train_x = np.array(train_x)
train_y = np.array(train_y)
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr = 0.0001, decay = 1e-6, momentum = 0.8, nesterov = True)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
print(model.summary())

model.fit(train_x, train_y, epochs=10, batch_size=632, validation_data = (test_x, test_y))




img_test_path = '../input/test/'
test_names = []
file_paths = []
for i in range(len(sample_f)):
    test_names.append(sample_f.loc[i][1])
    file_paths.append(img_test_path + str(int(sample_f.loc[i][0])) + '.jpg')

test_names = np.array(test_names)



test_img = []
for file_path in file_paths:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (img_size, img_size))
    
    test_img.append(img)
    
test_img = np.array(test_img)

predictions = np.array(model.predict(test_img))

import seaborn as sns
color = sns.color_palette()
index = []
for i in range(len(y)):
    index.append(i)
plt.figure(figsize=(24,8))
sns.pointplot(index[0:500] ,test_names[0:500],alpha=0.8, color=color[2])
sns.pointplot(index[0:500] ,predictions[0:500],alpha=0.8, color=color[1])
plt.show()


