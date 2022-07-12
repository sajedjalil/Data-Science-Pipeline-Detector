# kaggle kernel of AlGiLa
# I need to thanks the following contributors inspiring me on this kernel

# the function implementing k-fold come from the kernel provided by Finlay Liu 
# The idea to use Inception v3 come from the kernel provided by Ogurtsov it was in R, I wrote it in python

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


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
from keras.models import Model, load_model
from keras import applications
from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator

train_set = pd.read_csv('../input/train_labels.csv')
test_set = pd.read_csv('../input/sample_submission.csv')

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  #needed for Inception v3
    return img

train_img, test_img = [], []
for img_path in tqdm(train_set['name'].iloc[: ]):
    train_img.append(read_img('../input/train/' + str(img_path) + '.jpg'))
for img_path in tqdm(test_set['name'].iloc[: ]):
    test_img.append(read_img('../input/test/' + str(img_path) + '.jpg'))

train_img = np.array(train_img, np.float32) / 255
train_label = np.array(train_set['invasive'].iloc[: ])
test_img = np.array(test_img, np.float32) / 255

#Transfer learning da Inception V3 traino solo gli ultimi fully connected layers
img_rows, img_cols, img_channel = 224, 224, 3
base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',pooling='avg', input_shape=(img_rows, img_cols, img_channel))
print(base_model.summary())


#Adding custom Layers
add_model = Sequential()
add_model.add(Dense(1024, activation='relu',input_shape=base_model.output_shape[1:]))
add_model.add(Dropout(0.60))
add_model.add(Dense(1, activation='sigmoid'))
print(add_model.summary())

# creating the final model
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

# compile the model
opt = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = optimizers.SGD(lr = 0.0001, momentum = 0.8, nesterov = True)
# meglio la funzione sotto che lega il learning rate decay al monitor che si vuole
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              patience=5,
                              verbose=1,
                              factor=0.1,
                              cooldown=10,
                              min_lr=0.00001)  # funzione di kesar to reduce learning rate (new_lr = lr*factor)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
print(model.summary())

n_fold = 5
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
            rotation_range = 20,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            # zca_whitening = True,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest')
        datagen.fit(x_tr)

        model.fit_generator(datagen.flow(x_tr, y_tr, batch_size = 64),
            validation_data = (x_te, y_te), callbacks=[reduce_lr],
            steps_per_epoch = len(train_img) / 64, epochs = 45, verbose = 2)

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
test_set.to_csv('../submit.csv', index = None)