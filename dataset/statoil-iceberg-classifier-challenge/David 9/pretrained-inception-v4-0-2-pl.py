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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

train = pd.read_json("../input/train.json")
target_train=train['is_iceberg']
test = pd.read_json("../input/test.json")
#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3=(X_band_1+X_band_2)/2
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)
X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3=(X_band_test_1+X_band_test_2)/2
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)
#Import Keras.
from matplotlib import pyplot
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation,AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
from inception_v4 import inception_v4_base
from keras.models import load_model

def getInceptionv4Model():

    inputs_1 = Input((75, 75, 3))
    net = inception_v4_base(inputs_1)


    # Final pooling and prediction

    # 8 x 8 x 1536
    #net_old = AveragePooling2D((1,1), border_mode='valid')(net)

    # 1 x 1 x 1536
    #net_old = Dropout(0.2)(net_old)
    net_old = Flatten()(net)

    predictions_1 = Dense(output_dim=1001, activation='softmax')(net_old)

    model = Model(inputs_1, predictions_1, name='inception_v4')

    weights_path = 'imagenet_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)


    inputs = Input((75, 75, 3))

    # net_ft = AveragePooling2D((1,1), border_mode='valid')(net)
    # net_ft = Dropout(0.2)(net_ft)
    net_ft = Flatten()(net)


    '''
    net_ft= Dense(512, activation='relu', name='fc2')(net_ft)
    net_ft = Dropout(0.25)(net_ft)

    net_ft= Dense(256, activation='relu', name='fc3')(net_ft)
    net_ft = Dropout(0.25)(net_ft)
    '''

    predictions = Dense(1, activation='sigmoid')(net_ft)

    model = Model(inputs_1, predictions, name='inception_v4')


    sgd = SGD(lr=1e-2, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model



    '''
    base_model = inception_v4_model(img_rows=75, img_cols=75,num_classes=1)

    x = base_model.get_layer('average_pooling2d_16').output
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    # x = GlobalMaxPooling2D()(x)
    x= Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    x= Dense(256, activation='relu', name='fc3')(x)
    x = Dropout(0.25)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=predictions)

    sgd = SGD(lr=1e-2, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
    '''
getInceptionv4Model().summary()

#base CV structure
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=16, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

#Using K-fold Cross Validation.
import os

def myBaseCrossTrain(X_train, target_train):
    folds = list(StratifiedKFold(n_splits=200, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    y_valid_pred_log = 0.0*target_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        file_path = "%s_model_weights.hdf5"%j
        #os.remove(file_path)
        callbacks = get_callbacks(filepath=file_path, patience=5)
        galaxyModel=getInceptionv4Model()
        galaxyModel.fit(X_train_cv, y_train_cv,
                  batch_size=64,
                  epochs=180,
                  verbose=1,
                  validation_data=(X_holdout, Y_holdout),
                  callbacks=callbacks)

        #Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)

        #Getting Training Score
        score = galaxyModel.evaluate(X_train_cv, y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = galaxyModel.evaluate(X_holdout, Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Getting validation Score.
        pred_valid=galaxyModel.predict(X_holdout)
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting Test Scores
        temp_test=galaxyModel.predict(X_test)
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

    y_test_pred_log=y_test_pred_log/200

    print('\nLog Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    return y_test_pred_log

preds=myBaseCrossTrain(X_train, target_train)
#Submission for each day.
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=preds
submission.to_csv('sub_myinceptionv4_200fold.csv', index=False, float_format='%.6f')