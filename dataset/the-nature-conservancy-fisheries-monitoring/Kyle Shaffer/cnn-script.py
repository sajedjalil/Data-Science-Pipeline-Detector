# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os, pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils

from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import operator

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

np.random.seed(7)

labels = []
img_data = []

label_map = dict(zip([i for i in os.listdir('../input/train') if not i=='.DS_Store'], range(8)))

for d in os.listdir('../input/train'):
    print("Processing {}...".format(d))
    if d == '.DS_Store':
        continue
    for f in os.listdir(os.path.join('../input/train', d)):
        labels.append(label_map[d])
        im = load_img(os.path.join('../input/train', d, f))
        im_data = img_to_array(im)
        im_data.resize(100, 100, 3)
        img_data.append(im_data)
print("Unique number of labels:", len(set(labels)))
print("Number of images:", len(img_data))
print("Length of label map:", len(label_map))


def cnn():
    model = Sequential()
    model.add(Convolution2D(128,4,4, border_mode='valid', input_shape=(100,100,3), activation='relu'))
    model.add(Convolution2D(128,4,4))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X = np.array(img_data)
y = np.array(labels)

X = X.astype('float32')
y = y.astype('float32')
print("Data converted...")

X /= 255.

# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
y = np_utils.to_categorical(y)

model = cnn()
print("Model loaded...")
# model.fit(X_train, y_train, nb_epoch=10, batch_size=64, verbose=2)
model.fit(X, y, nb_epoch=10, batch_size=64, class_weight=None, verbose=1)

# preds = model.predict(X_test)
# print ("Log loss on hold-out set:", log_loss(y_test, preds))

test_im_names = []
test_imgs = []

for fname in os.listdir('../input/test_stg1'):
    if fname == '.DS_Store':
        continue
    test_im_names.append(fname)
    im = load_img(os.path.join('../input/test_stg1', fname))
    im_data = img_to_array(im)
    im_data.resize(100, 100, 3)
    test_imgs.append(im_data)
    
print("Number of images", len(test_imgs))
print("Image size:", test_imgs[0].shape)


X_test_final = np.array(test_imgs)
X_test_final.astype('float32')
X_test_final /= 255.

final_preds = model.predict(X_test_final)

prediction_df = pd.DataFrame(final_preds)
prediction_df.columns = sorted(label_map.items(), key=operator.itemgetter(1))
prediction_df['image'] = test_im_names
prediction_df.to_csv('submission.csv', encoding='utf8', index=False)

print("Done!")

