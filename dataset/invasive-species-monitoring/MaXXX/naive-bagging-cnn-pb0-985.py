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


import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn import model_selection
from sklearn import metrics


import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

train_set = pd.read_csv('../input/train_labels.csv')
test_set = pd.read_csv('../input/sample_submission.csv')

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    return img

train_img, test_img = [], []
for img_path in tqdm(train_set['name'].iloc[: ]):
    train_img.append(read_img('../input/train/' + str(img_path) + '.jpg'))
for img_path in tqdm(test_set['name'].iloc[: ]):
    test_img.append(read_img('../input/test/' + str(img_path) + '.jpg'))

train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255

def model_nn():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.65))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.55))
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.8, nesterov = True)
    model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
    print(model.summary())
    return model

n_fold = 8
kf = model_selection.KFold(n_splits = n_fold, shuffle = True)
eval_fun = metrics.roc_auc_score

def run_oof(tr_x, tr_y, te_x, kf):
    preds_train = np.zeros(len(tr_x), dtype = np.float)
    preds_test = np.zeros(len(te_x), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    for train_index, test_index in kf.split(tr_x):
        x_tr = tr_x[train_index]; x_te = tr_x[test_index]
        y_tr = tr_y[train_index]; y_te = tr_y[test_index]

        datagen = ImageDataGenerator(
            # featurewise_center = True,
            rotation_range = 30,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            # zca_whitening = True,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)

        model = model_nn()
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15, verbose=0, mode='auto')
        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 64),
            validation_data = (x_te, y_te), callbacks = [earlystop],
            steps_per_epoch = len(train_img) / 64, epochs = 1000, verbose = 2)

        train_loss.append(eval_fun(y_tr, model.predict(x_tr)[:, 0]))
        test_loss.append(eval_fun(y_te, model.predict(x_te)[:, 0]))

        preds_train[test_index] = model.predict(x_te)[:, 0]
        preds_test += model.predict(te_x)[:, 0]

        print('{0}: Train {1:0.5f} Val {2:0.5f}'.format(i, train_loss[-1], test_loss[-1]))
        i += 1

    print('Train: ', train_loss)
    print('Val: ', test_loss)
    print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(test_loss)))
    preds_test /= n_fold
    return preds_train, preds_test

train_pred, test_pred = run_oof(train_img, train_label, test_img, kf)

test_set['invasive'] = test_pred
test_set.to_csv('./submit.csv', index = None)