# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
from tqdm import tqdm

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "-l", "../input/test-tif"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

labels_df = pd.read_csv("../input/train.csv")
# print(labels_df.head())
#
# # Build list with unique labels
label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)
#
# # datas_df = pd.read_csv("data/sample_submission.csv")
#
# # Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
# # Display head
# # print(labels_df.head())
# labels_df.to_csv("data/train_list.csv", index=False)
train_df = labels_df

test_df = pd.read_csv("../input/sample_submission.csv")
test_df['tags'] = test_df['tags'].apply(lambda x: '')

# print(test_df)

x_train = []
x_test = []


for f in tqdm(train_df.values, miniters=1000):
    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f[0]))
    # print(img.size)
    # targets = np.zeros(17)
    # for t in tags.split(' '):
        # targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (32, 32)))
    # y_train.append(targets)

for f in tqdm(test_df.values, miniters=1000):
    img = cv2.imread('../input/test-jpg/{}.jpg'.format(f[0]))
    # targets = np.zeros(17)
    # for t in tags.split(' '):
        # targets[label_map[t]] = 1
    x_test.append(cv2.resize(img, (32, 32)))
    # y_train.append(targets)

x_train = np.array(x_train, np.float16) / 255.
x_test = np.array(x_test, np.float16) / 255.

y_train = train_df.values[:, range(2, 19)]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(x_valid, y_valid))

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# model.save('model_keras_cnn_epochs_3.h5')
# print("model save ok")

p_valid = model.predict(x_valid, batch_size=128)
print(fbeta_score(y_valid == 1, p_valid > 0.2, beta=2, average='samples'))
# print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

# p_test = model.predict(x_test, batch_size=128)
# preds = []
# for i in range(p_test.shape[0]):
#     preds.append(' '.join([label_list[j] for j in range(len(label_list)) if p_test[i, j]>0.5]))
# index_preds = list(map(lambda x: "test_" + str(x), range(len(preds))))
# print(len(index_preds))
# print(len(preds))
# preds_data = np.c_[index_preds, preds]
#print(preds_data.shape)
# np.savetxt('submission_keras_cnn_epochs_3.csv', preds_data,
#                delimiter=',', header='image_name,tags', comments='', fmt='%s')





